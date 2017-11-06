// Information about a single 1D block line
//
// Each section is a 4D array of positions, which after blocking is
// a 4D array of 8x8x8x8 blocks.  A block line is a 1D sequence of
// such blocks along one of the four dimensions.  Block lines are the
// basic unit of work in the endgame solver, as discussed in compute.h.
#pragma once

#include "pentago/end/history.h"
#include "pentago/base/section.h"
#include "pentago/utility/thread.h"
namespace pentago {
namespace end {

struct line_t {
  section_t section;
  uint8_t dimension : 2; // Which way the line points
  uint8_t length : 6; // Length in blocks
  Vector<uint8_t,3> block_base; // Block indices in the other 3 dimensions
  
  Vector<uint8_t,4> block(int i) const { // ith block in block units
    GEODE_ASSERT((unsigned)i<=(unsigned)length);
    return block_base.insert(i, dimension);
  }

  event_t line_event() const {
    return end::line_event(section,dimension,block_base);
  }

  event_t block_line_event(int i) const {
    return end::block_line_event(section,dimension,this->block(i));
  }
};

static_assert(sizeof(line_t)==12,"Line isn't packed enough");

// Dedicated type to catch errors
struct local_id_t {
  int id;

  local_id_t() : id(-1) {}
  explicit local_id_t(int id) : id(id) {}
  bool operator==(local_id_t o) const { return id==o.id; }
};

// A block and a local id
struct local_block_t {
  local_id_t local_id;
  section_t section;
  Vector<uint8_t,4> block;
};

ostream& operator<<(ostream& output, const line_t& line);

template<class T,int d> static inline size_t hash_value(const Vector<T,d>& v) {
  size_t h = 0;
  for (const auto& x : v) boost::hash_combine(h, x);
  return h;
}

static inline size_t hash_value(local_id_t i) { return i.id; }

}
}

namespace std {
template<> struct hash<pentago::end::local_id_t> {
  size_t operator()(pentago::end::local_id_t i) const { return i.id; }
};
}  // namespace std
