// Statistics

#include "stat.h"
#include <other/core/python/module.h>
#include <other/core/python/stl.h>
#include <other/core/utility/Timer.h>
namespace pentago {

using namespace other;
using std::make_pair;
using std::cout;
using std::endl;

uint64_t expanded_nodes;
uint64_t total_lookups;
uint64_t successful_lookups;
uint64_t distance_prunes;
double start_time;

static void clear_stats() {
  expanded_nodes = 0;
  total_lookups = 0;
  successful_lookups = 0;
  distance_prunes = 0;
  start_time = get_current_time();
}

void print_stats() {
  double time = get_current_time();
  cout << "expanded nodes = "<<expanded_nodes
       << ", total_lookups = "<<total_lookups
       << ", successful_lookups = "<<successful_lookups
       << ", distance_prunes = "<<distance_prunes
       << ", nodes/second = "<<expanded_nodes/(time-start_time)
       << endl;
}

static unordered_map<string,Ref<> > stats() {
  double time = get_current_time();
  unordered_map<string,Ref<> > stats;
  #define ST(stat) stats.insert(make_pair(string(#stat),steal_ref_check(to_python(stat))));
  ST(expanded_nodes)
  ST(total_lookups)
  ST(successful_lookups)
  ST(distance_prunes)
  stats.insert(make_pair(string("nodes/second"),steal_ref_check(to_python(expanded_nodes/(time-start_time)))));
  return stats;
}

}
using namespace pentago;
using namespace other::python;

void wrap_stat() {
  OTHER_FUNCTION(clear_stats)
  OTHER_FUNCTION(print_stats)
  OTHER_FUNCTION(stats)
}
