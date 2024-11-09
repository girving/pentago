// Add commas to large integers

#include "pentago/utility/large.h"
#include "pentago/utility/format.h"
#include "pentago/utility/range.h"
#include <cstdlib>
#include <cmath>
namespace pentago {

using std::abs;

string large(uint64_t x) {
  const string s = tfm::format("%d", x);
  const auto n = s.size();
  string r;
  for (const auto i : range(n)) {
    if (i && (n-i)%3==0)
      r.push_back(',');
    r.push_back(s[i]);
  }
  return r;
}

string large(int64_t x) {
  const auto s = large(uint64_t(abs(x)));
  return x<0 ? "-"+s : s;
}

}
