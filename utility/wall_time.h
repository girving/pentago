// Wall clock time
#pragma once

#include <sys/time.h>
namespace pentago {

static inline double wall_time() {
  timeval tv;
  gettimeofday(&tv,0);
  return (double)tv.tv_sec+1e-6*tv.tv_usec;
}

}
