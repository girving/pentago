// Information about a single 1D block line
#pragma once

#include <pentago/mpi/utility.h>
#include <pentago/section.h>
#include <pentago/thread.h>
namespace pentago {
namespace mpi {

struct line_t {
  section_t section;
  uint8_t dimension : 2; // Which way the line points
  uint8_t length : 6; // Length in blocks
  Vector<uint8_t,3> block_base; // Block indices in the other 3 dimensions
  
  Vector<uint8_t,4> block(int i) const { // ith block in block units
    OTHER_ASSERT((unsigned)i<=(unsigned)length);
    return block_base.insert(i,dimension);
  }

  event_t line_event() const {
    return pentago::mpi::line_event(section,dimension,block_base);
  }

  event_t block_line_event(int i) const {
    return pentago::mpi::block_line_event(section,dimension,this->block(i));
  }
};

BOOST_STATIC_ASSERT(sizeof(line_t)==12);

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

}
}
namespace other {
template<> struct is_packed_pod<pentago::mpi::local_id_t> : public mpl::true_ {};
}
