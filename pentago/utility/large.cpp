// Add commas to large integers

#include <pentago/utility/large.h>
#include <geode/utility/format.h>
namespace pentago {

using std::abs;
using namespace geode;

string large(uint64_t x) {
  const string s = format("%llu",x);
  const int n = int(s.size());
  string r;
  for (int i=0;i<n;i++) {
    if (i && (n-i)%3==0)
      r.push_back(',');
    r.push_back(s[i]);
  }
  return r;
}

string large(int64_t x) {
  const auto s = large(uint64_t(abs(x)));
  return x<0?"-"+s:s;
}

}
