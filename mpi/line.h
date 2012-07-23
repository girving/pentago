// Information about a single 1D block line
#pragma once

#include <pentago/section.h>
namespace pentago {
namespace mpi {

struct line_t {
  section_t section;
  int dimension; // Which way the line points
  int length; // Length in blocks
  Vector<int,3> block_base; // Block indices in the other 3 dimensions
  int node_step; // Number of nodes in all blocks except possibly the last 
  uint64_t block_id; // Unique id of first block (owners only)
  uint64_t node_offset; // Offset of block in global linear node space (owners only)
  
  Vector<int,4> block(int i) const {
    OTHER_ASSERT((unsigned)i<=(unsigned)length);
    return block_base.insert(i,dimension);
  }

  Vector<uint64_t,2> block_offsets(int i) const {
    OTHER_ASSERT((unsigned)i<=(unsigned)length);
    return vec(block_id+i,node_offset+node_step*i);
  }
};
BOOST_STATIC_ASSERT(sizeof(line_t)==48);

}
}
