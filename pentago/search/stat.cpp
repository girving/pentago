// Statistics

#include <pentago/search/stat.h>
#include <geode/python/wrap.h>
#include <geode/python/stl.h>
#include <geode/utility/time.h>
namespace pentago {

using namespace geode;
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
  start_time = get_time();
}

void print_stats() {
  double elapsed = get_time()-start_time;
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
using namespace pentago;
using namespace geode::python;

void wrap_stat() {
  GEODE_FUNCTION(clear_stats)
  GEODE_FUNCTION(print_stats)
#ifdef GEODE_PYTHON
  GEODE_FUNCTION(stats)
#endif
}
