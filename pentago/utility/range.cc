#include "pentago/utility/range.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/exceptions.h"
#include "pentago/utility/format.h"
#include <cstdint>
#include <string>
namespace pentago {

using std::min;

// Partition a loop into chunks based on the total number of threads.  Returns a half open interval.
template<class I,class TI> inline Range<I>
partition_loop(const I loop_steps, const TI threads, const TI thread) {
  static_assert(std::is_integral<I>::value && std::is_integral<TI>::value && sizeof(TI)<=sizeof(I));
  typedef std::make_unsigned_t<I> UI;
  typedef std::make_unsigned_t<TI> TUI;
  GEODE_ASSERT(threads>0 && TUI(thread)<TUI(threads));
  const I steps_per_thread = loop_steps / threads; // Round down, so some threads will get one more step
  const TI extra_steps = loop_steps % threads; // The first extra_steps threads will get one extra step
  const I start = steps_per_thread*thread + min(extra_steps, thread);
  const I end = start+steps_per_thread + (UI(thread)<UI(extra_steps));
  return Range<I>(start, end);
}

// Inverse of partition_loop: map an index to the thread that owns it
template<class I,class TI> inline TI
partition_loop_inverse(const I loop_steps, const TI threads, const I index) {
  static_assert(std::is_integral<I>::value && std::is_integral<TI>::value);
  typedef std::make_unsigned_t<I> UI;
  GEODE_ASSERT(threads>0 && UI(index)<UI(loop_steps));
  const I steps_per_thread = loop_steps / threads, // Round down, so some threads will get one more step
          extra_steps = loop_steps % threads, // The first extra_steps threads will get one extra step
          threshold = (steps_per_thread+1)*extra_steps; // Before here, all threads have an extra step
  return TI(     index<threshold  ? index/(steps_per_thread+1)
            : steps_per_thread ? extra_steps+(index-threshold)/steps_per_thread
                               : 0); // Only occurs if loop_steps==0
}

#define INSTANTIATE(I, TI) \
  template Range<I> partition_loop(const I, const TI, const TI); \
  template TI partition_loop_inverse(const I, const TI, const I);
INSTANTIATE(int, int)
INSTANTIATE(uint64_t, int)

static int parse_int(const string& s) {
  try {
    return std::stoi(s);
  } catch (const std::exception&) {
    THROW(ValueError, "invalid integer '%s'", s);
  }
}

Range<int> parse_range(const string& s, const int total) {
  GEODE_ASSERT(total >= 0);
  const auto colon = s.find(':');
  GEODE_ASSERT(colon != string::npos, "range must contain ':'");
  const auto lo_str = s.substr(0, colon);
  const auto hi_str = s.substr(colon + 1);
  const int lo = lo_str.empty() ? 0 : parse_int(lo_str);
  const int hi = hi_str.empty() ? total : parse_int(hi_str);
  GEODE_ASSERT(0 <= lo && lo <= hi && hi <= total,
               tfm::format("invalid range %d:%d for total %d", lo, hi, total));
  return {lo, hi};
}

}
