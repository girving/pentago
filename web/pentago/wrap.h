// Node.js wrapping utilities
// For example usage, see module.cpp.
#pragma once

#include <v8.h>
#include <node.h>
#include <node_buffer.h>
#include <pentago/base/section.h>
#include <geode/python/Ptr.h>
#include <geode/utility/Enumerate.h>
#include <geode/utility/format.h>
#include <geode/utility/range.h>
#include <geode/vector/Vector.h>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/noncopyable.hpp>
#include <boost/mpl/and.hpp>
namespace pentago {
namespace node {

using namespace v8;
using namespace geode;
namespace mpl = boost::mpl;
using std::string;
using std::vector;
template<class T> struct Wrapper;

#define PN_CLASS(name,make) \
  typedef name Self; \
  const auto sname = String::NewSymbol(#name); \
  Wrapper<Self>::template_ = Persistent<FunctionTemplate>::New( \
    FunctionTemplate::New(Wrapper<Self>::wrapped_constructor<make>)); \
  Wrapper<Self>::template_->SetClassName(sname); \
  Wrapper<Self>::template_->InstanceTemplate()->SetInternalFieldCount(1); \
  Wrapper<Self>::Finish finisher(exports,sname);

#define PN_METHOD(name) \
  static_assert(boost::is_same<decltype(finisher),Wrapper<Self>::Finish>::value,""); \
  Wrapper<Self>::template_->PrototypeTemplate()->Set(String::NewSymbol(#name),FunctionTemplate::New( \
    wrapped_method<Self,decltype(&Self::name),&Self::name>)->GetFunction());

#define PN_FUNCTION(name) \
  exports->Set(String::NewSymbol(#name),FunctionTemplate::New( \
    wrapped_function<decltype(&name),name>)->GetFunction());

// Conversion

template<class T> struct is_numeric : public
  mpl::and_<mpl::not_<boost::is_same<T,bool>>,
            mpl::or_<boost::is_integral<T>,
                     boost::is_floating_point<T>>> {};

static inline Handle<Primitive> to_js(unit) {
  return Undefined();
}

template<class T> static inline typename boost::enable_if<boost::is_same<T,bool>,Handle<Boolean>>::type to_js(T x) {
  return Boolean::New(x);
}

template<class T> static inline typename boost::enable_if<is_numeric<T>,Handle<Number>>::type to_js(T x) {
  return Number::New(x);
}

static inline Handle<String> to_js(const string& x) {
  return String::New(x.c_str());
}

template<class T> static inline Handle<v8::Object> to_js(const Ref<T>& x) {
  typedef typename boost::remove_const<T>::type MT;
  GEODE_ASSERT(!Wrapper<MT>::constructor.IsEmpty(),format("type %s has not been wrapped",typeid(T).name()));
  Wrapper<MT>::constructor_hack = x.const_cast_();
  const auto js = Wrapper<MT>::constructor->NewInstance(0,0);
  Wrapper<MT>::constructor_hack.clear();
  return js;
}

template<class T> static inline Handle<v8::Array> to_js(const vector<T>& xs);
template<class T0,class T1> static inline Handle<v8::Array> to_js(const Tuple<T0,T1>& t);
template<class T,int d> static inline Handle<v8::Array> to_js(const Vector<T,d>& x);

static inline Handle<v8::Array> to_js(const section_t x) {
  return to_js(x.counts);
}

template<class T> static inline Handle<v8::Array> to_js(const vector<T>& xs) {
  int n = 0;
  auto v = v8::Array::New();
  for (const auto& x : xs)
    v->Set(n++,to_js(x));
  return v;
}

template<class T0,class T1> static inline Handle<v8::Array> to_js(const Tuple<T0,T1>& t) {
  auto v = v8::Array::New();
  v->Set(0,to_js(t.x));
  v->Set(1,to_js(t.y));
  return v;
}

template<class T,int d> static inline Handle<v8::Array> to_js(const Vector<T,d>& x) {
  auto v = v8::Array::New();
  for (int i=0;i<d;i++)
    v->Set(i,to_js(x[i]));
  return v;
}

template<class T,class enable=void> struct FromJS;
template<class T> struct FromJS<const T> : public FromJS<T> {};

template<class T> static inline T from_js(const Local<Value>& x) {
  return FromJS<T>::convert(x);
}

template<class T> struct FromJS<T,typename boost::enable_if<is_numeric<T>>::type> {
  static T convert(const Local<Value>& x) {
    const auto n = Local<Number>::Cast(x);
    if (x.IsEmpty())
      throw TypeError(format("expected %s",typeid(T).name()));
    const double d = n->Value();
    if (T(d)!=d)
      throw TypeError(format("expected %s, got %g",typeid(T).name(),d));
    return T(d);
  }
};

template<class T> struct FromJS<T&,typename boost::enable_if<boost::is_base_of<geode::Object,T>>::type> {
  static T& convert(const Local<Value>& x) {
    typedef typename boost::remove_const<T>::type MT;
    if (x->IsObject()) {
      const auto o = x->ToObject();
      if (Wrapper<MT>::template_->HasInstance(o)) {
        const Wrapper<MT>& self = *::node::ObjectWrap::Unwrap<Wrapper<MT>>(x->ToObject());
        if (!boost::is_const<T>::value && !self.mutable_)
          throw format("expected mutable %s, got const",typeid(MT).name());
        return self.self;
      }
    }
    throw TypeError(format("expected object, type %s",typeid(MT).name()));
  }
};

template<class T0,class T1> struct FromJS<Tuple<T0,T1>> { static Tuple<T0,T1> convert(const Local<Value>& x) {
  if (x->IsArray()) {
    const auto a = Local<v8::Array>::Cast(x);
    if (a->Length() == 2)
      return tuple(from_js<T0>(a->Get(0)),
                   from_js<T1>(a->Get(1)));
  }
  throw TypeError(format("expected length 2 tuple, types %s, %s",typeid(T0).name(),typeid(T1).name()));
}};

template<> struct FromJS<RawArray<const uint8_t>> { static RawArray<const uint8_t> convert(const Local<Value>& x) {
  if (!::node::Buffer::HasInstance(x))
    throw TypeError("expected Buffer");
  return RawArray<const uint8_t>(::node::Buffer::Length(x),(const uint8_t*)::node::Buffer::Data(x));
}};

template<class T,int d> struct FromJS<Vector<T,d>> { static Vector<T,d> convert(const Local<Value>& x) {
  if (!x->IsArray())
    throw TypeError(format("expected length %d vector",d));
  const auto a = Local<v8::Array>::Cast(x);
  if (a->Length() != d)
    throw TypeError(format("expected length %d vector, got length %d",d,a->Length()));
  Vector<T,d> v;
  for (int i=0;i<d;i++)
    v[i] = from_js<T>(a->Get(i));
  return v;
}};

template<> struct FromJS<section_t> { static section_t convert(const Local<Value>& x) {
  return section_t(from_js<Vector<Vector<uint8_t,2>,4>>(x));
}};

// Class wrapping

template<class T> struct Wrapper : public ::node::ObjectWrap {
  static_assert(!boost::is_const<T>::value,"");
  typedef T Self;
  const geode::Ref<T> self;
  const bool mutable_;

  Wrapper(T& self, const bool mutable_)
    : self(ref(self))
    , mutable_(mutable_) {}

  static Persistent<FunctionTemplate> template_;
  static Persistent<Function> constructor;

  // If nonnull, constructor calls make this object and ignore their arguments.  This is very ugly.
  static Ptr<T> constructor_hack;
  static bool constructor_hack_mutable;

  template<geode::Ref<T>(*factory)(const Arguments&)> static Handle<Value> wrapped_constructor(const Arguments& args);

  struct Finish : public boost::noncopyable {
    Handle<v8::Object>& exports;
    const Local<String>& sname;
    Finish(Handle<v8::Object>& exports, const Local<String>& sname) : exports(exports), sname(sname) {}
    ~Finish() {
      Wrapper::constructor = Persistent<Function>::New(Wrapper::template_->GetFunction());
      if (!boost::is_same<typename T::Base,geode::Object>::value) {
        GEODE_ASSERT(!Wrapper<typename T::Base>::template_.IsEmpty());
        Wrapper::template_->Inherit(Wrapper<typename T::Base>::template_);
      }
      exports->Set(sname,Wrapper::constructor);
    }
  };
};

template<class T> Persistent<FunctionTemplate> Wrapper<T>::template_;
template<class T> Persistent<Function> Wrapper<T>::constructor;
template<class T> Ptr<T> Wrapper<T>::constructor_hack;
template<class T> bool Wrapper<T>::constructor_hack_mutable;

template<class T> template<Ref<T>(*factory)(const Arguments&)> Handle<Value>
Wrapper<T>::wrapped_constructor(const Arguments& args) {
  HandleScope scope;
  if (args.IsConstructCall()) {
    // Invoked as constructor
    const bool mutable_ = !Wrapper<T>::constructor_hack || Wrapper<T>::constructor_hack_mutable;
    Ptr<T> self = Wrapper<T>::constructor_hack;
    if (!self) {
      try {
        const geode::Ref<T> r = factory(args);
        self = r;
      } catch (const exception& e) {
        ThrowException(Exception::Error(String::New(e.what())));
        return scope.Close(Undefined());
      }
    }
    auto wrapper = new Wrapper<T>(*self,mutable_);
    wrapper->Wrap(args.This());
    return args.This();
  } else {
    // Invoked as plain function, turn into construct call
    vector<Local<Value>> argv;
    for (const int i : range(args.Length()))
      argv.push_back(args[i]);
    return scope.Close(Wrapper<T>::constructor->NewInstance(argv.size(),&argv[0]));
  }
}

template<class M> struct enumerate_args;
template<class F,class T> struct enumerate_args<F T::*> { typedef Types<> type; };
template<class R,class... Args> struct enumerate_args<R(*)(Args...)> : public Enumerate<Args...> {};
template<class R,class T,class... Args> struct enumerate_args<R(T::*)(Args...)> : public Enumerate<Args...> {};
template<class R,class T,class... Args> struct enumerate_args<R(T::*)(Args...)const> : public Enumerate<Args...> {};

template<class A> static inline auto convert_item(const Arguments& args)
  -> decltype(from_js<typename A::type>(args[A::index])) {
  return from_js<typename A::type>(args[A::index]);
}

// Fields
template<class T,class F,class IA> static inline const F&
invoke(const T& self, F T::*field, const Arguments& args, IA, const bool mutable_) {
  return self.*field;
}

// Static methods
template<class T,class R,class... Args,class... IArgs> static inline R
invoke(const T& self, R (*method)(Args...), const Arguments& args, Types<IArgs...>, const bool mutable_) {
  return method(convert_item<IArgs>(args)...);
}

// Const methods
template<class T,class R,class... Args,class... IArgs> static inline R
invoke(const T& self, R (T::*method)(Args...) const, const Arguments& args, Types<IArgs...>, const bool mutable_) {
  return (self.*method)(convert_item<IArgs>(args)...);
}

// Nonconst methods
template<class T,class R,class... Args,class... IArgs> static inline R
invoke(T& self, R (T::*method)(Args...), const Arguments& args, Types<IArgs...>, const bool mutable_) {
  if (!mutable_)
    throw TypeError(format("nonconst method called on immutable object of type %s",typeid(T).name()));
  return (self.*method)(convert_item<IArgs>(args)...);
}

template<class T,class M,M method> static Handle<Value> wrapped_method(const Arguments& args) {
  HandleScope scope;
  typedef typename enumerate_args<M>::type Args;
  if (args.Length() != Args::size) {
    ThrowException(Exception::Error(to_js(format("expected %d arguments, got %d",Args::size,args.Length()))));
    return scope.Close(Undefined());
  }
  try {
    typedef typename boost::remove_const<T>::type MT;
    const auto self = ::node::ObjectWrap::Unwrap<Wrapper<MT>>(args.This());
    return scope.Close(to_js(invoke(*self->self,method,args,Args(),self->mutable_)));
  } catch (const exception& e) {
    ThrowException(Exception::Error(String::New(e.what())));
    return scope.Close(Undefined());
  }
}

// Free functions

template<class F,F func> static Handle<Value> wrapped_function(const Arguments& args) {
  HandleScope scope;
  typedef typename enumerate_args<F>::type Args;
  if (args.Length() != Args::size) {
    ThrowException(Exception::Error(to_js(format("expected %d arguments, got %d",Args::size,args.Length()))));
    return scope.Close(Undefined());
  }
  try {
    return scope.Close(to_js(invoke(unit(),func,args,Args(),false)));
  } catch (const exception& e) {
    ThrowException(Exception::Error(String::New(e.what())));
    return scope.Close(Undefined());
  }
}

}
}
