// Join a list of strings with a separator
#pragma once

#include <string>
namespace pentago {

template<class A> static inline string join(const string& sep, const A& list) {
  string result;
  bool first = true;
  for (const string& s : list) {
    if (!first)
      result += sep;
    first = false;
    result += s;
  }
  return result;
}

}
