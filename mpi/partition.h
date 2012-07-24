// Partitioning of sections and lines for MPI purposes
#pragma once

#include <pentago/mpi/line.h>
#include <pentago/section.h>
#include <other/core/geometry/Box.h>
#include <other/core/structure/Hashtable.h>
#include <vector>
namespace pentago {
namespace mpi {

using std::vector;

// Compute all sections that root depends, organized by slice.
// Only 35 slices are returned, since computing slice 36 is unnecessary.
vector<Array<const section_t>> descendent_sections(section_t root);

// A set of 1D block lines
struct lines_t {
  // Section information
  section_t section;
  Vector<int,4> shape;

  // Line information
  int dimension; // Line dimension
  int length; // Line length in blocks
  Box<Vector<int,3>> blocks; // Ranges of blocks along the other three dimensions
  int count; // Number of lines in this range = blocks.volume()
  int node_step; // Number of nodes in all blocks except possibly those at the end of lines
  int line_size; // All lines in this line range have the same size

  // Running total information, meaningful for owners only
  uint64_t block_id; // Unique id of first block (others are indexed consecutively)
  uint64_t node_offset; // Total number of nodes in all blocks in lines before this
};
BOOST_STATIC_ASSERT(sizeof(lines_t)==88);

// Given a set of sections, distribute all lines amongst a number of processors
struct partition_t : public Object {
  OTHER_DECLARE_TYPE

  const int ranks;
  const int block_size;
  const int slice; // Common number of stones in each section
  const Array<const section_t> sections;
  const Array<const lines_t> owner_lines, other_lines;
  const Hashtable<section_t,int> first_owner_line;
  const Array<const Vector<int,2>> owner_starts, other_starts;
  const Array<const uint64_t> owner_work, other_work; // Empty unless save_work is true
  const double owner_excess, total_excess; // Initialized only if verbose
  const uint64_t total_blocks, total_nodes;
  const int max_rank_blocks; // Maximum number of blocks owned by a rank

protected:
  partition_t(const int ranks, const int block_size, const int slice, Array<const section_t> sections, bool save_work=false);
public:
  ~partition_t();

  // Estimate memory usage
  uint64_t memory_usage() const;

  // (first block id, global node offset) for a given rank
  Vector<uint64_t,2> rank_offsets(int rank) const;

  // List all lines belonging to a given rank
  Array<line_t> rank_lines(int rank, bool owned) const;

  // Count all lines belonging to a given rank
  uint64_t rank_count_lines(int rank, bool owned) const;

  // Find the rank which owns a given block
  int block_to_rank(section_t section, Vector<int,4> block) const;

  // Find the line that owns a given block
  Vector<int,2> block_to_line(section_t section, Vector<int,4> block) const;

  // (block_id, global node offset) for a given block
  Vector<uint64_t,2> block_offsets(section_t section, Vector<int,4> block) const;

private:
  // Can the remaining work fit within the given bound?
  template<bool record> static bool fit(RawArray<uint64_t> work_nodes, RawArray<uint64_t> work_penalties, RawArray<const lines_t> lines, const uint64_t bound, RawArray<Vector<int,2>> starts=(RawArray<Vector<int,2>>()));

  // Divide a set of lines between processes
  static Array<const Vector<int,2>> partition_lines(RawArray<uint64_t> work_nodes, RawArray<uint64_t> work_penalties, RawArray<const lines_t> lines);
};

}
}
