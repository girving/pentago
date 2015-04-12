// Node.js wrapping utilities
// For example usage, see module.cpp.
#pragma once

#include <v8.h>
#include <node.h>
#include <node_buffer.h>
#include <node_object_wrap.h>
#include <pentago/base/section.h>
#include <pentago/high/index.h>
#include <geode/python/Ptr.h>
#include <geode/utility/Enumerate.h>
#include <geode/utility/format.h>
#include <geode/utility/mpl.h>
#include <geode/utility/range.h>
#include <geode/utility/type_traits.h>
#include <geode/vector/Vector.h>
namespace pentago {
namespace node {

using namespace v8;
using namespace geode;
using std::string;
using std::vector;
template<class T> struct Wrapper;

// Wrap a class
#define PN_CLASS(name,make) \
  typedef name Self; \
  Isolate* const iso = Isolate::GetCurrent(); \
  const auto sname = new_symbol(iso,#name); \
  Wrapper<Self>::template_.reset(new Persistent<FunctionTemplate>(iso, \
    FunctionTemplate::New(iso,Wrapper<Self>::wrapped_constructor<make>))); \
  value(iso,*Wrapper<Self>::template_)->SetClassName(sname); \
  value(iso,*Wrapper<Self>::template_)->InstanceTemplate()->SetInternalFieldCount(1); \
  Wrapper<Self>::Finish finisher(iso,exports,sname);

// Wrap a class method
#define PN_METHOD(name) \
  PN_METHOD_2(name,name)

// Wrap a class method with a different name
#define PN_METHOD_2(name,method) \
  static_assert(is_same<decltype(finisher),Wrapper<Self>::Finish>::value,""); \
  value(iso,*Wrapper<Self>::template_)->PrototypeTemplate()->Set(new_symbol(iso,#name),FunctionTemplate::New(iso, \
    wrapped_method<Self,decltype(&Self::method),&Self::method>)->GetFunction());

// Wrap a free function
#define PN_FUNCTION(name) { \
  Isolate* const iso = Isolate::GetCurrent(); \
  exports->Set(new_symbol(iso,#name),FunctionTemplate::New(iso, \
    wrapped_function<decltype(&name),name>)->GetFunction()); }

// Utilities

static inline Local<String> new_symbol(Isolate* iso, const string& s) {
  return String::NewFromUtf8(iso,s.c_str(),String::kInternalizedString);
}

template<class T> static Local<T> value(Isolate* iso, const Persistent<T>& persist) {
  return Local<T>::New(iso,persist);
}

// Conversion

template<class T> struct is_numeric : public
  mpl::and_<mpl::not_<is_same<T,bool>>,
            mpl::or_<is_integral<T>,
                     is_floating_point<T>>> {};

static inline Handle<Primitive> to_js(Isolate* iso, Unit) {
  return Undefined(iso);
}

template<class T> static inline typename enable_if<is_same<T,bool>,Handle<Boolean>>::type to_js(Isolate* iso, T x) {
  return Boolean::New(iso,x);
}

template<class T> static inline typename enable_if<is_numeric<T>,Handle<Number>>::type to_js(Isolate* iso, T x) {
  const double d(x);
  if (!is_floating_point<T>::value && T(d)!=x)
    throw ValueError(format("Can't safely convert %s to javascript, would degrade to %s",str(x),str(d)));
  return Number::New(iso,x);
}

static inline Handle<String> to_js(Isolate* iso, const char* x) {
  return String::NewFromOneByte(iso,(const uint8_t*)x);
}

static inline Handle<String> to_js(Isolate* iso, const string& x) {
  return String::NewFromOneByte(iso,(const uint8_t*)x.c_str());
}

template<class T> static inline Handle<v8::Object> to_js(Isolate* iso, const Ref<T>& x) {
  typedef typename remove_const<T>::type MT;
  GEODE_ASSERT(Wrapper<MT>::constructor,format("type %s has not been wrapped",typeid(T).name()));
  Wrapper<MT>::constructor_hack = x.const_cast_();
  const auto js = value(iso,*Wrapper<MT>::constructor)->NewInstance(0,0);
  Wrapper<MT>::constructor_hack.clear();
  return js;
}

template<class T> static inline Handle<v8::Array> to_js(Isolate* iso, const vector<T>& xs);
template<class T0,class T1> static inline Handle<v8::Array> to_js(Isolate* iso, const Tuple<T0,T1>& t);
template<class T,int d> static inline Handle<v8::Array> to_js(Isolate* iso, const Vector<T,d>& x);

static inline Handle<v8::Array> to_js(Isolate* iso, const section_t x) {
  return to_js(iso,x.counts);
}

static inline Handle<v8::Object> to_js(Isolate* iso, const compact_blob_t x) {
  auto o = v8::Object::New(iso);
  o->Set(to_js(iso,"offset"),to_js(iso,x.offset()));
  o->Set(to_js(iso,"size"),to_js(iso,x.size));
  return o;
}

template<class T> static inline Handle<v8::Array> to_js(Isolate* iso, const vector<T>& xs) {
  int n = 0;
  auto v = v8::Array::New(iso);
  for (const auto& x : xs)
    v->Set(n++,to_js(iso,x));
  return v;
}

template<class T0,class T1> static inline Handle<v8::Array> to_js(Isolate* iso, const Tuple<T0,T1>& t) {
  auto v = v8::Array::New(iso);
  v->Set(0,to_js(iso,t.x));
  v->Set(1,to_js(iso,t.y));
  return v;
}

template<class T,int d> static inline Handle<v8::Array> to_js(Isolate* iso, const Vector<T,d>& x) {
  auto v = v8::Array::New(iso);
  for (int i=0;i<d;i++)
    v->Set(i,to_js(iso,x[i]));
  return v;
}

template<class K,class V> static inline Handle<v8::Object> to_js(Isolate* iso, const Hashtable<K,V>& x) {
  auto o = v8::Object::New(iso);
  for (auto& y : x)
    o->Set(to_js(iso,y.x),to_js(iso,y.y));
  return o;
}

template<class T,class enable=void> struct FromJS;
template<class T> struct FromJS<const T> : public FromJS<T> {};

template<class T> static inline auto from_js(Isolate* iso, const Local<Value>& x)
  -> decltype(FromJS<T>::convert(iso,x)) {
  return FromJS<T>::convert(iso,x);
}

template<class T> struct FromJS<T,typename enable_if<is_numeric<T>>::type> {
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

template<class T> struct FromJS<T&,typename enable_if<is_base_of<geode::Object,T>>::type> {
  static T& convert(Isolate* iso, const Local<Value>& x) {
    typedef typename remove_const<T>::type MT;
    if (x->IsObject()) {
      const auto o = x->ToObject();
      if (value(iso,*Wrapper<MT>::template_)->HasInstance(o)) {
        const Wrapper<MT>& self = *::node::ObjectWrap::Unwrap<Wrapper<MT>>(x->ToObject());
        if (!is_const<T>::value && !self.mutable_)
          throw format("expected mutable %s, got const",typeid(MT).name());
        return self.self;
      }
    }
    throw TypeError(format("expected object, type %s",typeid(MT).name()));
  }
};
template<class T> struct FromJS<T&,typename disable_if<is_base_of<geode::Object,T>>::type>
  : public FromJS<T> {};

template<class T> struct FromJS<Ref<T>> {
  static Ref<T> convert(Isolate* iso, const Local<Value>& x) {
    return ref(from_js<T&>(iso,x));
  }
};

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

template<class T0,class T1> struct FromJS<Tuple<T0,T1>> { static Tuple<T0,T1> convert(Isolate* iso, const Local<Value>& x) {
  if (x->IsArray()) {
    const auto a = Local<v8::Array>::Cast(x);
    if (a->Length() == 2)
      return tuple(from_js<T0>(iso,a->Get(0)),
                   from_js<T1>(iso,a->Get(1)));
  }
  throw TypeError(format("expected length 2 tuple, types %s, %s",typeid(T0).name(),typeid(T1).name()));
}};

template<> struct FromJS<RawArray<uint8_t>> { static RawArray<uint8_t> convert(Isolate* iso, const Local<Value>& x) {
  if (!::node::Buffer::HasInstance(x))
    throw TypeError("expected Buffer");
  return RawArray<uint8_t>(::node::Buffer::Length(x),(uint8_t*)::node::Buffer::Data(x));
}};
template<> struct FromJS<RawArray<const uint8_t>> : public FromJS<RawArray<uint8_t>> {};

template<class T,int d> struct FromJS<Vector<T,d>> { static Vector<T,d> convert(Isolate* iso, const Local<Value>& x) {
  if (!x->IsArray())
    throw TypeError(format("expected length %d vector",d));
  const auto a = Local<v8::Array>::Cast(x);
  if (a->Length() != d)
    throw TypeError(format("expected length %d vector, got length %d",d,a->Length()));
  Vector<T,d> v;
  for (int i=0;i<d;i++)
    v[i] = from_js<T>(iso,a->Get(i));
  return v;
}};

template<> struct FromJS<section_t> { static section_t convert(Isolate* iso, const Local<Value>& x) {
  return section_t(from_js<Vector<Vector<uint8_t,2>,4>>(iso,x));
}};

// Class wrapping

typedef FunctionCallbackInfo<v8::Value> Arguments;

template<class T> struct Wrapper : public ::node::ObjectWrap {
  static_assert(!is_const<T>::value,"");
  typedef T Self;
  const geode::Ref<T> self;
  const bool mutable_;

  Wrapper(T& self, const bool mutable_)
    : self(ref(self))
    , mutable_(mutable_) {}

  static scoped_ptr<Persistent<FunctionTemplate>> template_;
  static scoped_ptr<Persistent<Function>> constructor;

  // If nonnull, constructor calls make this object and ignore their arguments.  This is very ugly.
  static Ptr<T> constructor_hack;
  static bool constructor_hack_mutable;

  template<geode::Ref<T>(*factory)(const Arguments&)> static void wrapped_constructor(const Arguments& args);

  struct Finish : public Noncopyable {
    Isolate* const iso;
    Handle<v8::Object>& exports;
    const Local<String>& sname;
    Finish(Isolate* iso, Handle<v8::Object>& exports, const Local<String>& sname)
      : iso(iso), exports(exports), sname(sname) {}
    ~Finish() {
      Local<FunctionTemplate> template_ = value(iso,*Wrapper::template_);
      Wrapper::constructor.reset(new Persistent<Function>(iso,template_->GetFunction()));
      if (!is_same<typename T::Base,geode::Object>::value) {
        GEODE_ASSERT(Wrapper<typename T::Base>::template_);
        template_->Inherit(value(iso,*Wrapper<typename T::Base>::template_));
      }
      exports->Set(sname,value(iso,*Wrapper::constructor));
    }
  };
};

template<class T> scoped_ptr<Persistent<FunctionTemplate>> Wrapper<T>::template_;
template<class T> scoped_ptr<Persistent<Function>> Wrapper<T>::constructor;
template<class T> Ptr<T> Wrapper<T>::constructor_hack;
template<class T> bool Wrapper<T>::constructor_hack_mutable;

template<class T> template<Ref<T>(*factory)(const Arguments&)> void
Wrapper<T>::wrapped_constructor(const Arguments& args) {
  Isolate* const iso = args.GetIsolate();
  if (args.IsConstructCall()) {
    // Invoked as constructor
    const bool mutable_ = !Wrapper<T>::constructor_hack || Wrapper<T>::constructor_hack_mutable;
    Ptr<T> self = Wrapper<T>::constructor_hack;
    if (!self) {
      try {
        const geode::Ref<T> r = factory(args);
        self = r;
      } catch (const exception& e) {
        iso->ThrowException(Exception::Error(String::NewFromUtf8(iso,e.what())));
        return args.GetReturnValue().Set(Undefined(iso));
      }
    }
    auto wrapper = new Wrapper<T>(*self,mutable_);
    wrapper->Wrap(args.This());
    return args.GetReturnValue().Set(args.This());
  } else {
    // Invoked as plain function, turn into construct call
    vector<Local<Value>> argv;
    for (const int i : range(args.Length()))
      argv.push_back(args[i]);
    return args.GetReturnValue().Set(value(iso,*Wrapper<T>::constructor)->NewInstance(argv.size(),&argv[0]));
  }
}

template<class M> struct enumerate_args;
template<class F,class T> struct enumerate_args<F T::*> { typedef Types<> type; };
template<class R,class... Args> struct enumerate_args<R(*)(Args...)> : public Enumerate<Args...> {};
template<class R,class T,class... Args> struct enumerate_args<R(T::*)(Args...)> : public Enumerate<Args...> {};
template<class R,class T,class... Args> struct enumerate_args<R(T::*)(Args...)const> : public Enumerate<Args...> {};

template<class A> static inline auto convert_item(Isolate* iso, const Arguments& args)
  -> decltype(from_js<typename A::type>(iso,args[A::index])) {
  return from_js<typename A::type>(iso,args[A::index]);
}

// Fields
template<class T,class F,class IA> static inline const F&
invoke(const T& self, F T::*field, const Arguments& args, IA, const bool mutable_) {
  return self.*field;
}

// Static methods
template<class T,class R,class... Args,class... IArgs> static inline R
invoke(const T& self, R (*method)(Args...), const Arguments& args, Types<IArgs...>, const bool mutable_) {
  Isolate* iso = args.GetIsolate();
  return method(convert_item<IArgs>(iso,args)...);
}

// Const methods
template<class T,class R,class... Args,class... IArgs> static inline R
invoke(const T& self, R (T::*method)(Args...) const, const Arguments& args, Types<IArgs...>, const bool mutable_) {
  Isolate* iso = args.GetIsolate();
  return (self.*method)(convert_item<IArgs>(iso,args)...);
}

// Nonconst methods
template<class T,class R,class... Args,class... IArgs> static inline R
invoke(T& self, R (T::*method)(Args...), const Arguments& args, Types<IArgs...>, const bool mutable_) {
  if (!mutable_)
    throw TypeError(format("nonconst method called on immutable object of type %s",typeid(T).name()));
  Isolate* iso = args.GetIsolate();
  return (self.*method)(convert_item<IArgs>(iso,args)...);
}

template<class T,class M,M method> static void wrapped_method(const Arguments& args) {
  Isolate* const iso = args.GetIsolate();
  typedef typename enumerate_args<M>::type Args;
  if (args.Length() != Args::size) {
    iso->ThrowException(Exception::Error(to_js(iso,format("expected %d arguments, got %d",Args::size,args.Length()))));
    return args.GetReturnValue().Set(Undefined(iso));
  }
  try {
    typedef typename remove_const<T>::type MT;
    const auto self = ::node::ObjectWrap::Unwrap<Wrapper<MT>>(args.This());
    return args.GetReturnValue().Set(to_js(iso,invoke(*self->self,method,args,Args(),self->mutable_)));
  } catch (const exception& e) {
    iso->ThrowException(Exception::Error(String::NewFromUtf8(iso,e.what())));
    return args.GetReturnValue().Set(Undefined(iso));
  }
}

// Free functions

template<class F,F func> static void wrapped_function(const Arguments& args) {
  Isolate* const iso = args.GetIsolate();
  typedef typename enumerate_args<F>::type Args;
  if (args.Length() != Args::size) {
    iso->ThrowException(Exception::Error(to_js(iso,format("expected %d arguments, got %d",Args::size,args.Length()))));
    return args.GetReturnValue().Set(Undefined(iso));
  }
  try {
    return args.GetReturnValue().Set(to_js(iso,invoke(unit,func,args,Args(),false)));
  } catch (const exception& e) {
    iso->ThrowException(Exception::Error(String::NewFromUtf8(iso,e.what())));
    return args.GetReturnValue().Set(Undefined(iso));
  }
}

}
}
