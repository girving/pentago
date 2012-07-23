// Add commas to large integers

#include <pentago/utility/large.h>
#include <other/core/utility/format.h>
namespace pentago {

using namespace other;

string large(uint64_t x) {
  string s = format("%llu",x);
  int n = s.size();
  string r;
  for (int i=0;i<n;i++) {
    if (i && (n-i)%3==0)
      r.push_back(',');
    r.push_back(s[i]);
  }
  return r;
}

}
