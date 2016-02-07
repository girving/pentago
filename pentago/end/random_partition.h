// Random partitioning of lines and blocks across processes
//
// random_partition_t is the lightest possible weight partitioning scheme:
// the only shared state is a single uint128_t key (besides the list of
// sections).  Internally, blocks and lines are assigned to different processors
// using deterministic counter mode random number generators, relying on the
// large of large numbers to spread work uniformly across the ranks.  Since there
// is quite a lot of work to spread out, this works quite well.
#pragma once

#include <pentago/end/partition.h>
#include <pentago/end/line.h>
#include <pentago/base/section.h>
#include <geode/math/uint128.h>
namespace pentago {
namespace end {

struct sheaf_t;

// Given a set of sections, distribute all lines and blocks amongst a number of processors
struct random_partition_t : public partition_t {
  GEODE_DECLARE_TYPE(GEODE_EXPORT)
  typedef partition_t Base;

  const uint128_t key; // Key for random number generation
  const int total_lines;
  const Array<const sheaf_t> sheafs; // One per section per dimension
  const Hashtable<Tuple<section_t,uint8_t>,int> sheaf_id; // Map from section,dimension to sheaf index

protected:
  GEODE_EXPORT random_partition_t(const uint128_t key, const int ranks, const sections_t& sections);
public:
  ~random_partition_t();

  // Define partition_t interface
  uint64_t memory_usage() const override;
  uint64_t rank_count_lines(const int rank) const override;
  Array<const line_t> rank_lines(const int rank) const override;
  Array<const local_block_t> rank_blocks(const int rank) const override;
  Vector<uint64_t,2> rank_counts(const int rank) const override; // Expensive, since it calls rank_blocks
  Tuple<int,local_id_t> find_block(const section_t section, const Vector<uint8_t,4> block) const override;
  Tuple<section_t,Vector<uint8_t,4>> rank_block(const int rank, const local_id_t local_id) const override;

private:

  // A random ordering of the lines
  line_t nth_line(const int n) const;

  // Which dimension owns a given block
  uint8_t owner_dimension(const section_t section, const Vector<uint8_t,4> block) const;
};

}
}
