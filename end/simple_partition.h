// Partitioning of sections and lines for MPI purposes
#pragma once

#include <pentago/end/partition.h>
#include <pentago/end/line.h>
#include <pentago/end/sections.h>
#include <other/core/geometry/Box.h>
#include <other/core/structure/Hashtable.h>
#include <vector>
namespace pentago {
namespace end {

struct chunk_t;
using std::vector;

// Given a set of sections, distribute all lines amongst a number of processors
struct simple_partition_t : public partition_t {
  OTHER_DECLARE_TYPE(OTHER_EXPORT)
  typedef partition_t Base;

  const Array<const chunk_t> owner_lines, other_lines;
  const Hashtable<section_t,int> first_owner_line;
  const Array<const Vector<int,2>> owner_starts, other_starts;
  const Array<const uint64_t> owner_work, other_work; // Empty unless save_work is true
  const double owner_excess, total_excess; // Initialized only if verbose
  const int max_rank_blocks; // Maximum number of blocks owned by a rank
  const uint64_t max_rank_nodes; // Maximum number of nodes owned by a rank

protected:
  OTHER_EXPORT simple_partition_t(const int ranks, const sections_t& sections, bool save_work=false);
public:
  ~simple_partition_t();

  // Define partition_t interface
  uint64_t memory_usage() const;
  uint64_t rank_count_lines(const int rank) const override;
  Array<const line_t> rank_lines(const int rank) const override;
  Array<const local_block_t> rank_blocks(const int rank) const override;
  Vector<uint64_t,2> rank_counts(const int rank) const override;
  Tuple<int,local_id_t> find_block(const section_t section, const Vector<uint8_t,4> block) const override;
  Tuple<section_t,Vector<uint8_t,4>> rank_block(const int rank, const local_id_t local_id) const override;

private:
  friend void simple_partition_test();

  // (first block id, global node offset) for a given rank
  Vector<uint64_t,2> rank_offsets(int rank) const;

  // List all lines belonging to a given rank
  Array<line_t> rank_lines(int rank, bool owned) const;

  // Count all lines belonging to a given rank
  uint64_t rank_count_lines(int rank, bool owned) const;

  // Find the rank which owns a given block
  int block_to_rank(section_t section, Vector<uint8_t,4> block) const;

  // Find the line that owns a given block
  Vector<int,2> block_to_line(section_t section, Vector<uint8_t,4> block) const;

  // Find the global id for a given block
  uint64_t block_to_id(section_t section, Vector<uint8_t,4> block) const;

  // Can the remaining work fit within the given bound?
  template<bool record> static bool fit(RawArray<uint64_t> work_nodes, RawArray<uint64_t> work_penalties, RawArray<const chunk_t> lines, const uint64_t bound, RawArray<Vector<int,2>> starts=(RawArray<Vector<int,2>>()));

  // Divide a set of lines between processes
  static Array<const Vector<int,2>> partition_lines(RawArray<uint64_t> work_nodes, RawArray<uint64_t> work_penalties, RawArray<const chunk_t> lines, const int line_count);
};

}
}
