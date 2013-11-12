// Wall clock time
#pragma once

#include <geode/utility/safe_bool.h>
#include <sys/time.h>
namespace pentago {

struct wall_time_t {
  int64_t us; // Microseconds since January 1, 1970, or relative microseconds

  wall_time_t()
    : us(0) {}

  explicit wall_time_t(int64_t us)
    : us(us) {}

  double seconds() const {
    return 1e-6*us;
  }

  operator SafeBool() const {
    return safe_bool(us!=0);
  }

  wall_time_t& operator+=(wall_time_t t) {
    us += t.us; return *this;
  }

  wall_time_t& operator-=(wall_time_t t) {
    us -= t.us; return *this;
  }

  wall_time_t operator-(wall_time_t t) const {
    return wall_time_t(us-t.us);
  }

  bool operator==(wall_time_t t) const { return us == t.us; }
  bool operator!=(wall_time_t t) const { return us != t.us; }
  bool operator< (wall_time_t t) const { return us <  t.us; }
  bool operator<=(wall_time_t t) const { return us <= t.us; }
  bool operator> (wall_time_t t) const { return us >  t.us; }
  bool operator>=(wall_time_t t) const { return us >= t.us; }
};

static inline wall_time_t wall_time() {
  timeval tv;
  gettimeofday(&tv,0);
  return wall_time_t(tv.tv_sec*1000000+tv.tv_usec);
}

}
