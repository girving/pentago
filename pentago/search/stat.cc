// Statistics

#include "pentago/search/stat.h"
#include "pentago/utility/array.h"
#include "pentago/utility/wall_time.h"
#include "pentago/utility/log.h"
namespace pentago {

using std::make_pair;

uint64_t total_expanded_nodes;
uint64_t expanded_nodes[37];
uint64_t total_lookups;
uint64_t successful_lookups;
STAT_DETAIL(uint64_t lookup_detail[37], successful_lookup_detail[37];)
uint64_t distance_prunes;
static wall_time_t start_time;

void clear_stats() {
  total_expanded_nodes = 0;
  asarray(expanded_nodes).fill(0);
  total_lookups = 0;
  successful_lookups = 0;
  STAT_DETAIL(
    asarray(lookup_detail).fill(0);
    asarray(successful_lookup_detail).fill(0);
  )
  distance_prunes = 0;
  start_time = wall_time();
}

void print_stats() {
  const auto elapsed = wall_time()-start_time;
  string s = format("expanded nodes = %d (", total_expanded_nodes);
  int found = 0;
  for (int d=36;d>0;d--)
    if (expanded_nodes[d])
      s += format("%s%d", found++?" ":"", expanded_nodes[d]);
  s += ")";
  if (total_lookups) {
    s += format(", total lookups = %d", total_lookups);
    STAT_DETAIL(
      s += " (";
      int found = 0;
      for (int d=36;d>0;d--)
        if (lookup_detail[d])
          s += format("%s%d:%d", found++?" ":"", d, lookup_detail[d]);
      s += ")";
    )
  }
  if (successful_lookups) {
    s += format(", successful lookups = %d", successful_lookups);
    STAT_DETAIL(
      s += " (";
      int found = 0;
      for (int d=36;d>0;d--)
        if (successful_lookup_detail[d])
          s += format("%s%d:%d", found++?" ":"", d, successful_lookup_detail[d]);
      s += ")";
    )
  }
  if (distance_prunes) s += format(", distance prunes = %d", distance_prunes);
  s += format(", elapsed time = %g s", elapsed.seconds());
  s += format(", speed = %d nodes/s", uint64_t(total_expanded_nodes/elapsed.seconds()));
  slog(s);
}

#ifdef GEODE_PYTHON
static unordered_map<string,Ref<> > stats() {
  double time = get_time();
  unordered_map<string,Ref<> > stats;
  #define ST(stat) stats.insert(make_pair(string(#stat),steal_ref_check(to_python(stat))));
  ST(total_lookups)
  ST(successful_lookups)
  ST(distance_prunes)
  stats.insert(make_pair(string("nodes/second"),steal_ref_check(to_python(uint64_t(total_expanded_nodes/(time-start_time))))));
  return stats;
}
#endif

}
