// Tiny XML/SVG tag emission helpers shared by the website generators
#pragma once

#include "pentago/utility/format.h"
#include <string>
#include <vector>
namespace pentago {

using std::string;
using std::tuple;
using std::vector;

static inline string start_tag(const string& name, const vector<tuple<string,string>>& attrs,
                               const bool close = false) {
  string tag = "<" + name;
  for (const auto& [k,v] : attrs)
    if (v.size())
      tag += tfm::format(" %s=\"%s\"", k, v);
  tag += close ? "/>" : ">";
  return tag;
}

static inline string close_tag(const string& name) {
  return tfm::format("</%s>", name);
}

static inline string tag(const string& name, const vector<tuple<string,string>>& attrs,
                         const string& body = "") {
  string tag = start_tag(name, attrs, body.empty());
  if (body.size()) tag += body + close_tag(name);
  return tag;
}

static inline string svg(const int width, const int height, const string& body) {
  return tag("svg", {{"viewBox", tfm::format("0 0 %d %d", width, height)},
                     {"xmlns", "http://www.w3.org/2000/svg"}}, body);
}

}  // namespace pentago
