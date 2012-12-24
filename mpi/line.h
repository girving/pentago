// Information about a single 1D block line
#pragma once

#include <pentago/mpi/utility.h>
#include <pentago/section.h>
#include <pentago/thread.h>
namespace pentago {
namespace mpi {

struct line_t {
  section_t section;
  unsigned dimension : 2; // Which way the line points
  uint8_t length; // Length in blocks
  Vector<uint8_t,3> block_base; // Block indices in the other 3 dimensions
  int node_step; // Number of nodes in all blocks except possibly the last 
  uint64_t block_id; // Unique id of first block (owners only)
  uint64_t node_offset; // Offset of block in global linear node space (owners only)
  
  Vector<uint8_t,4> block(int i) const { // ith block in block units
    OTHER_ASSERT((unsigned)i<=(unsigned)length);
    return block_base.insert(i,dimension);
  }

  Vector<uint64_t,2> block_offsets(int i) const {
    OTHER_ASSERT((unsigned)i<=(unsigned)length);
    return vec(block_id+i,node_offset+node_step*i);
  }

  event_t line_event() const {
    return pentago::mpi::line_event(section,dimension,block_base);
  }

  event_t block_line_event(int i) const {
    return pentago::mpi::block_line_event(section,dimension,this->block(i));
  }
};

BOOST_STATIC_ASSERT(sizeof(line_t)==40);

ostream& operator<<(ostream& output, const line_t& line);

}
}
