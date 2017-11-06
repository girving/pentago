// Stream based convertion from T to str
#pragma once

#include <sstream>
#include <tuple>
namespace pentago {

using std::string;
using std::tuple;
using std::get;

static inline string str() {
  return string();
}

template<class T> string str(const T& x) {
  std::ostringstream os;
  os << x;
  return os.str();
}

static inline const string& str(const string& x) {
  return x;
}

static inline string str(const char* x) {
  return x;
}

template<class A,class B> static inline string str(const tuple<A,B>& t) {
  std::ostringstream os;
  os << '(' << get<0>(t) << ',' << get<1>(t) << ')';
  return os.str();
}

template<class A,class B,class C> static inline string str(const tuple<A,B,C>& t) {
  std::ostringstream os;
  os << '(' << get<0>(t) << ',' << get<1>(t) << ',' << get<2>(t) << ')';
  return os.str();
}

}
