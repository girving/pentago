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

uint64_t total_expanded_nodes;
uint64_t expanded_nodes[37];
uint64_t total_lookups;
uint64_t successful_lookups;
uint64_t distance_prunes;
double start_time;

static void clear_stats() {
  total_expanded_nodes = 0;
  memset(expanded_nodes,0,sizeof(expanded_nodes));
  total_lookups = 0;
  successful_lookups = 0;
  distance_prunes = 0;
  start_time = get_current_time();
}

void print_stats() {
  double elapsed = get_current_time()-start_time;
  cout << "expanded nodes = "<<total_expanded_nodes<<" (";
  bool found = false;
  for (int d=36;d>0;d--)
    if (expanded_nodes[d]) {
      cout << (found?" ":"")<<expanded_nodes[d];
      found = true;
    }
  cout << ")";
  if (total_lookups) cout << ", total lookups = "<<total_lookups;
  if (successful_lookups) cout << ", successful lookups = "<<successful_lookups;
  if (distance_prunes) cout << ", distance prunes = "<<distance_prunes;
  cout << ", elapsed time = "<<elapsed<<" s";
  cout << ", speed = "<<uint64_t(total_expanded_nodes/elapsed)<<" nodes/s"<<endl;
}

static unordered_map<string,Ref<> > stats() {
  double time = get_current_time();
  unordered_map<string,Ref<> > stats;
  #define ST(stat) stats.insert(make_pair(string(#stat),steal_ref_check(to_python(stat))));
  ST(expanded_nodes)
  ST(total_lookups)
  ST(successful_lookups)
  ST(distance_prunes)
  stats.insert(make_pair(string("nodes/second"),steal_ref_check(to_python(uint64_t(total_expanded_nodes/(time-start_time))))));
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
