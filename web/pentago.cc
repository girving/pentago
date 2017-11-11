// Node.js bindings for a tiny bit of pentago

#include <v8.h>
#include <node.h>
#include <node_buffer.h>
#include <node_object_wrap.h>
#include <pentago/base/section.h>
#include <pentago/data/async_block_cache.h>
#include <pentago/end/sections.h>
#include <pentago/high/board.h>
#include <pentago/high/index.h>
#include <pentago/mid/midengine.h>
#include <unordered_map>
namespace pentago {
namespace node {
namespace {

using namespace v8;
using namespace pentago::end;
using std::make_shared;
using std::string;
using std::unordered_map;
using std::vector;
template<class T> struct Wrapper;

// Declare wrapped classes
template<class T> struct pn_valid { static constexpr bool value = false; };
template<class T> struct pn_valid<const T> : public pn_valid<T> {};
#define PN_TYPE(T) template<> struct pn_valid<T> { static constexpr bool value = true; };
PN_TYPE(high_board_t)
PN_TYPE(async_block_cache_t)

// Wrap a class
#define PN_CLASS(name, make) \
  typedef name Self; \
  Isolate* const iso = Isolate::GetCurrent(); \
  const auto sname = new_symbol(iso, #name); \
  Wrapper<Self>::template_.reset(new Persistent<FunctionTemplate>(iso, \
      FunctionTemplate::New(iso, Wrapper<Self>::wrapped_constructor<make>))); \
  value(iso, *Wrapper<Self>::template_)->SetClassName(sname); \
  value(iso, *Wrapper<Self>::template_)->InstanceTemplate()->SetInternalFieldCount(1); \
  Wrapper<Self>::Finish finisher(iso, exports, sname);

// Wrap a class method
#define PN_METHOD(name) \
  PN_METHOD_2(name, name)

// Wrap a class method with a different name
#define PN_METHOD_2(name, method) \
  static_assert(std::is_same<decltype(finisher),Wrapper<Self>::Finish>::value); \
  value(iso, *Wrapper<Self>::template_)->PrototypeTemplate()->Set( \
      new_symbol(iso, #name), FunctionTemplate::New(iso, \
          wrapped_method<Self,decltype(&Self::method),&Self::method>));

// Wrap a free function
#define PN_FUNCTION(name) { \
  Isolate* const iso = Isolate::GetCurrent(); \
  exports->Set(new_symbol(iso, #name), FunctionTemplate::New(iso, \
    wrapped_function<decltype(&name),name>)->GetFunction()); }

// Utilities

Local<String> new_symbol(Isolate* iso, const string& s) {
  return String::NewFromUtf8(iso, s.c_str(), String::kInternalizedString);
}

template<class T> Local<T> value(Isolate* iso, const Persistent<T>& persist) {
  return Local<T>::New(iso, persist);
}

// Conversion

template<class T> struct is_numeric { static constexpr bool value =
    !std::is_same<T,bool>::value && (std::is_integral<T>::value || std::is_floating_point<T>::value); };

Handle<Primitive> to_js(Isolate* iso, unit_t) {
  return Undefined(iso);
}

template<class T> std::enable_if_t<std::is_same<T,bool>::value,Handle<Boolean>>
to_js(Isolate* iso, T x) {
  return Boolean::New(iso,x);
}

template<class T> std::enable_if_t<is_numeric<T>::value,Handle<Number>>
to_js(Isolate* iso, T x) {
  const double d(x);
  if (T(d) != x)
    throw ValueError(format("Can't safely convert %s to javascript, would degrade to %s",
                            str(x), str(d)));
  return Number::New(iso, x);
}

Handle<String> to_js(Isolate* iso, const char* x) {
  return String::NewFromOneByte(iso, (const uint8_t*)x);
}

Handle<String> to_js(Isolate* iso, const string& x) {
  return String::NewFromOneByte(iso, (const uint8_t*)x.c_str());
}

template<class T> Handle<v8::Object> to_js(Isolate* iso, const shared_ptr<T>& x) {
  typedef std::remove_const_t<T> MT;
  GEODE_ASSERT(Wrapper<MT>::constructor, format("type %s has not been wrapped", typeid(T).name()));
  Wrapper<MT>::constructor_hack = std::const_pointer_cast<MT>(x);
  const auto js = value(iso, *Wrapper<MT>::constructor)->NewInstance(0,0);
  Wrapper<MT>::constructor_hack.reset();
  return js;
}

template<class T> Handle<v8::Array> to_js(Isolate* iso, const vector<T>& xs);
template<class T0,class T1> Handle<v8::Array> to_js(Isolate* iso, const tuple<T0,T1>& t);
template<class T,int d> Handle<v8::Array> to_js(Isolate* iso, const Vector<T,d>& x);

Handle<v8::Array> to_js(Isolate* iso, const section_t x) {
  return to_js(iso, x.counts);
}

Handle<v8::Object> to_js(Isolate* iso, const high_board_t& x) {
  return to_js(iso, make_shared<high_board_t>(x));
}

Handle<v8::Object> to_js(Isolate* iso, const compact_blob_t x) {
  auto o = v8::Object::New(iso);
  o->Set(to_js(iso, "offset"), to_js(iso, x.offset()));
  o->Set(to_js(iso, "size"), to_js(iso, x.size));
  return o;
}

template<class T> Handle<v8::Array> to_js(Isolate* iso, const vector<T>& xs) {
  int n = 0;
  auto v = v8::Array::New(iso);
  for (const auto& x : xs)
    v->Set(n++, to_js(iso, x));
  return v;
}

template<class T0,class T1> Handle<v8::Array> to_js(Isolate* iso, const tuple<T0,T1>& t) {
  const auto& [x, y] = t;
  auto v = v8::Array::New(iso);
  v->Set(0, to_js(iso, x));
  v->Set(1, to_js(iso, y));
  return v;
}

template<class T,int d> Handle<v8::Array> to_js(Isolate* iso, const Vector<T,d>& x) {
  auto v = v8::Array::New(iso);
  for (const int i : range(d))
    v->Set(i, to_js(iso, x[i]));
  return v;
}

template<class K,class V> Handle<v8::Object> to_js(Isolate* iso, const unordered_map<K,V>& x) {
  auto o = v8::Object::New(iso);
  for (const auto& p : x)
    o->Set(to_js(iso, p.first), to_js(iso, p.second));
  return o;
}

template<class T,class enable=void> struct FromJS;
template<class T> struct FromJS<const T> : public FromJS<T> {};

template<class T> decltype(auto) from_js(Isolate* iso, const Local<Value>& x) {
  return FromJS<T>::convert(iso, x);
}

template<class T> struct FromJS<T,std::enable_if_t<is_numeric<T>::value>> {
  static T convert(Isolate* iso, const Local<Value>& x) {
    const auto n = Local<Number>::Cast(x);
    if (x.IsEmpty())
      throw TypeError(format("expected %s",typeid(T).name()));
    const double d = n->Value();
    if (T(d)!=d)
      throw TypeError(format("expected %s, got %g",typeid(T).name(),d));
    return T(d);
  }
};

template<class T> struct FromJS<shared_ptr<T>> {
  static const shared_ptr<std::remove_const_t<T>>& convert(Isolate* iso, const Local<Value>& x) {
    typedef std::remove_const_t<T> MT;
    if (x->IsObject()) {
      const auto o = x->ToObject();
      if (value(iso, *Wrapper<MT>::template_)->HasInstance(o)) {
        const Wrapper<MT>& self = *::node::ObjectWrap::Unwrap<Wrapper<MT>>(x->ToObject());
        if (!std::is_const<T>::value && !self.mutable_)
          throw format("expected mutable %s, got const", typeid(MT).name());
        return self.self;
      }
    }
    throw TypeError(format("expected object, type %s", typeid(MT).name()));
  }
};

template<class T> struct FromJS<T&,std::enable_if_t<pn_valid<T>::value>> {
  static T& convert(Isolate* iso, const Local<Value>& x) {
    return *FromJS<shared_ptr<T>>::convert(iso, x);
  }
};

template<class T> struct FromJS<T&,std::enable_if_t<!pn_valid<T>::value>> : public FromJS<T> {};

template<class T> struct FromJS<vector<T>> {
  static vector<T> convert(Isolate* iso, const Local<Value>& x) {
    if (!x->IsArray())
      throw TypeError(format("expected array of %s",typeid(T).name()));
    const auto a = Local<v8::Array>::Cast(x);
    const int n = a->Length();
    vector<T> r;
    for (const int i : range(n))
      r.push_back(from_js<T>(iso,a->Get(i)));
    return r;
  }
};

template<class T0,class T1> struct FromJS<tuple<T0,T1>> {
  static tuple<T0,T1> convert(Isolate* iso, const Local<Value>& x) {
    if (x->IsArray()) {
      const auto a = Local<v8::Array>::Cast(x);
      if (a->Length() == 2)
        return make_tuple(from_js<T0>(iso, a->Get(0)),
                          from_js<T1>(iso, a->Get(1)));
    }
    throw TypeError(format("expected length 2 tuple, types %s, %s", typeid(T0).name(),
                    typeid(T1).name()));
  }
};

template<> struct FromJS<RawArray<uint8_t>> {
  static RawArray<uint8_t> convert(Isolate* iso, const Local<Value>& x) {
    if (!::node::Buffer::HasInstance(x))
      throw TypeError("expected Buffer");
    return RawArray<uint8_t>(::node::Buffer::Length(x), (uint8_t*)::node::Buffer::Data(x));
  }
};
template<> struct FromJS<RawArray<const uint8_t>> : public FromJS<RawArray<uint8_t>> {};

template<> struct FromJS<high_board_t> {
  static high_board_t convert(Isolate* iso, const Local<Value>& x) {
    return *FromJS<shared_ptr<const high_board_t>>::convert(iso, x);
  }
};
template<> struct FromJS<RawArray<const high_board_t>> : public FromJS<vector<high_board_t>> {};

template<> struct FromJS<const block_cache_t&> : public FromJS<const async_block_cache_t&> {};

template<class T,int d> struct FromJS<Vector<T,d>> {
  static Vector<T,d> convert(Isolate* iso, const Local<Value>& x) {
    if (!x->IsArray())
      throw TypeError(format("expected length %d vector",d));
    const auto a = Local<v8::Array>::Cast(x);
    if (a->Length() != d)
      throw TypeError(format("expected length %d vector, got length %d", d, a->Length()));
    Vector<T,d> v;
    for (const int i : range(d))
      v[i] = from_js<T>(iso, a->Get(i));
    return v;
  }
};

template<> struct FromJS<section_t> { static section_t convert(Isolate* iso, const Local<Value>& x) {
  return section_t(from_js<Vector<Vector<uint8_t,2>,4>>(iso, x));
}};

// Class wrapping

typedef FunctionCallbackInfo<v8::Value> Arguments;

template<class T> struct Wrapper : public ::node::ObjectWrap {
  static_assert(!std::is_const<T>::value);
  typedef T Self;
  const shared_ptr<T> self;
  const bool mutable_;

  Wrapper(const shared_ptr<T>& self, const bool mutable_)
    : self(self), mutable_(mutable_) {}

  static unique_ptr<Persistent<FunctionTemplate>> template_;
  static unique_ptr<Persistent<Function>> constructor;

  // If nonnull, constructor calls make this object and ignore their arguments.  This is very ugly.
  static shared_ptr<T> constructor_hack;
  static bool constructor_hack_mutable;

  template<shared_ptr<T>(*factory)(const Arguments&)> static void
  wrapped_constructor(const Arguments& args);

  struct Finish : public boost::noncopyable {
    Isolate* const iso;
    Handle<v8::Object>& exports;
    const Local<String>& sname;
    Finish(Isolate* iso, Handle<v8::Object>& exports, const Local<String>& sname)
      : iso(iso), exports(exports), sname(sname) {}
    ~Finish() {
      Local<FunctionTemplate> template_ = value(iso, *Wrapper::template_);
      Wrapper::constructor.reset(new Persistent<Function>(iso, template_->GetFunction()));
      exports->Set(sname, value(iso, *Wrapper::constructor));
    }
  };
};

template<class T> unique_ptr<Persistent<FunctionTemplate>> Wrapper<T>::template_;
template<class T> unique_ptr<Persistent<Function>> Wrapper<T>::constructor;
template<class T> shared_ptr<T> Wrapper<T>::constructor_hack;
template<class T> bool Wrapper<T>::constructor_hack_mutable;

template<class T> template<shared_ptr<T>(*factory)(const Arguments&)> void
Wrapper<T>::wrapped_constructor(const Arguments& args) {
  Isolate* const iso = args.GetIsolate();
  if (args.IsConstructCall()) {
    // Invoked as constructor
    const bool mutable_ = !Wrapper<T>::constructor_hack || Wrapper<T>::constructor_hack_mutable;
    auto self = Wrapper<T>::constructor_hack;
    if (!self) {
      try {
        self = factory(args);
      } catch (const exception& e) {
        iso->ThrowException(Exception::Error(String::NewFromUtf8(iso, e.what())));
        return args.GetReturnValue().Set(Undefined(iso));
      }
    }
    auto wrapper = new Wrapper<T>(self, mutable_);
    wrapper->Wrap(args.This());
    return args.GetReturnValue().Set(args.This());
  } else {
    // Invoked as plain function, turn into construct call
    vector<Local<Value>> argv;
    for (const int i : range(args.Length()))
      argv.push_back(args[i]);
    return args.GetReturnValue().Set(value(iso, *Wrapper<T>::constructor)->NewInstance(
        argv.size(), &argv[0]));
  }
}

// Convert the ith argument to type A
template<size_t i,class A> decltype(auto) convert_item(Isolate* iso, const Arguments& args) {
  return from_js<A>(iso, args[i]);
}

// Make an index_sequence with n = function arity
template<class F> struct arity;
template<class F,class T> struct arity<F T::*> { static constexpr int value = 0; };
template<class R,class... A> struct arity<R(*)(A...)> { static constexpr int value = sizeof...(A); };
template<class R,class T,class... A> struct arity<R(T::*)(A...)> {
  static constexpr int value = sizeof...(A); };
template<class R,class T,class... A> struct arity<R(T::*)(A...)const> {
  static constexpr int value = sizeof...(A); };
template<class F> using arity_indices = std::make_index_sequence<arity<F>::value>;

// Fields
template<class T,class F,class indices> const F&
invoke(const T& self, F T::*field, const Arguments& args, indices, const bool mutable_) {
  return self.*field;
}

// Static methods
template<class T,class R,size_t... indices, class... Args> R
invoke(const T& self, R (*method)(Args...), const Arguments& args, std::index_sequence<indices...>,
       const bool mutable_) {
  return method(convert_item<indices,Args>(args.GetIsolate(), args)...);
}

// Const methods
template<class T,class R,size_t... indices,class... Args> R
invoke(const T& self, R (T::*method)(Args...) const, const Arguments& args,
       std::index_sequence<indices...>, const bool mutable_) {
  return (self.*method)(convert_item<indices,Args>(args.GetIsolate(), args)...);
}

// Nonconst methods
template<class T,class R,size_t... indices, class... Args> R
invoke(T& self, R (T::*method)(Args...), const Arguments& args, std::index_sequence<indices...>,
       const bool mutable_) {
  if (!mutable_)
    throw TypeError(format("nonconst method called on immutable object of type %s", typeid(T).name()));
  return (self.*method)(convert_item<indices,Args>(args.GetIsolate(), args)...);
}

template<class T,class M,M method> void wrapped_method(const Arguments& args) {
  Isolate* const iso = args.GetIsolate();
  typedef arity_indices<M> indices;
  if (args.Length() != indices::size()) {
    iso->ThrowException(Exception::Error(to_js(
        iso, format("expected %d arguments, got %d", indices::size(), args.Length()))));
    return args.GetReturnValue().Set(Undefined(iso));
  }
  try {
    typedef std::remove_const_t<T> MT;
    const auto self = ::node::ObjectWrap::Unwrap<Wrapper<MT>>(args.This());
    return args.GetReturnValue().Set(to_js(
        iso, invoke(*self->self, method, args, indices(), self->mutable_)));
  } catch (const exception& e) {
    iso->ThrowException(Exception::Error(String::NewFromUtf8(iso, e.what())));
    return args.GetReturnValue().Set(Undefined(iso));
  }
}

// Free functions

template<class F,F func> void wrapped_function(const Arguments& args) {
  Isolate* const iso = args.GetIsolate();
  typedef arity_indices<F> indices;
  if (args.Length() != indices::size()) {
    iso->ThrowException(Exception::Error(to_js(
        iso, format("expected %d arguments, got %d", indices::size(), args.Length()))));
    return args.GetReturnValue().Set(Undefined(iso));
  }
  try {
    return args.GetReturnValue().Set(to_js(iso, invoke(unit, func, args, indices(), false)));
  } catch (const exception& e) {
    iso->ThrowException(Exception::Error(String::NewFromUtf8(iso, e.what())));
    return args.GetReturnValue().Set(Undefined(iso));
  }
}

// Pentago specific

shared_ptr<high_board_t> make_board(const Arguments& args) {
  return args.Length() ? make_shared<high_board_t>(high_board_t::parse(*String::Utf8Value(args[0])))
                       : make_shared<high_board_t>(0, false);
}

shared_ptr<sections_t> make_sections(const Arguments& args) {
  throw TypeError("sections_t can only be made via descendent_sections()");
}

shared_ptr<async_block_cache_t> make_async(const Arguments& args) {
  if (args.Length() < 1 || !args[0]->IsNumber())
    throw TypeError("async_block_cache_t: expected one numeric memory_limit argument");
  return make_shared<async_block_cache_t>(uint64_t(args[0]->NumberValue()));
}

shared_ptr<supertensor_index_t> make_index(const Arguments& args) {
  Isolate* iso = args.GetIsolate();
  if (args.Length() < 1)
    throw TypeError(format("supertensor_index_t: expected one sections_t argument, got %d arguments",
                           args.Length()));
  return make_shared<supertensor_index_t>(from_js<const shared_ptr<const sections_t>>(iso, args[0]));
}

void init(Handle<v8::Object> exports) {
  {
    PN_CLASS(high_board_t, make_board)
    PN_METHOD(name)
    PN_METHOD(done)
    PN_METHOD(count)
    PN_METHOD(turn)
    PN_METHOD(middle)
    PN_METHOD(moves)
    PN_METHOD(value)
    PN_METHOD(immediate_value)
    PN_METHOD_2(toString, name)
  } {
    PN_CLASS(sections_t, make_sections)
  } {
    PN_CLASS(async_block_cache_t, make_async)
    PN_METHOD(board_block)
    PN_METHOD(contains)
    PN_METHOD(set)
  } {
    PN_CLASS(supertensor_index_t, make_index)
    PN_METHOD(blob_location)
    PN_METHOD(block_location)
  }
  PN_FUNCTION(descendent_sections)
  PN_FUNCTION(midsolve_workspace_memory_usage)
  PN_FUNCTION(high_midsolve)
  PN_FUNCTION(init_threads)
}

}  // namespace
}  // namespace node
}  // namespace pentago

NODE_MODULE(pentago, pentago::node::init)
