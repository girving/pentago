// Statistics

#include "stat.h"
#include <other/core/python/module.h>
namespace pentago {

using namespace other;

uint64_t expanded_nodes;
uint64_t total_lookups;
uint64_t successful_lookups;
uint64_t distance_prunes;

static void clear_stats() {
  expanded_nodes = 0;
  total_lookups = 0;
  successful_lookups = 0;
  distance_prunes = 0;
}

static PyObject* stats() {
  PyObject* stats = PyDict_New();
  #define ST(stat) PyDict_SetItemString(stats,#stat,PyInt_FromLong(stat));
  ST(expanded_nodes)
  ST(total_lookups)
  ST(successful_lookups)
  ST(distance_prunes)
  return stats;
}

}
using namespace pentago;
using namespace other::python;

void wrap_stat() {
  OTHER_FUNCTION(clear_stats)
  OTHER_FUNCTION(stats)
}
