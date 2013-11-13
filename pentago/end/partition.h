// Abstract base class for partitioning of lines and blocks
#pragma once

#include <pentago/end/line.h>
#include <pentago/end/sections.h>
#include <geode/structure/Hashtable.h>
namespace pentago {
namespace end {

// Given a set of sections, distribute all blocks amongst a number of processors
struct block_partition_t : public Object {
  GEODE_DECLARE_TYPE(GEODE_EXPORT)

  const int ranks;
  const Ref<const sections_t> sections;

protected:
  block_partition_t(const int ranks, const sections_t& sections);
public:
  virtual ~block_partition_t();

  // Estimate memory usage
  virtual uint64_t memory_usage() const = 0;

  // List all blocks belonging to a given rank
  virtual Array<const local_block_t> rank_blocks(const int rank) const = 0;

  // Count blocks and nodes belonging to a given rank, returning (blocks,nodes).
  // WARNING: This may be even more expensive than listing the blocks explicitly with rank_blocks.
  virtual Vector<uint64_t,2> rank_counts(const int rank) const = 0;

  // Given a block, find its owner's rank and the local id on that rank
  virtual Tuple<int,local_id_t> find_block(const section_t section, const Vector<uint8_t,4> block) const = 0;

  // Find the block with given rank and local id
  virtual Tuple<section_t,Vector<uint8_t,4>> rank_block(const int rank, const local_id_t local_id) const = 0;
};

// Given a set of sections, distribute all lines amongst a number of processors
struct partition_t : public block_partition_t {
  GEODE_DECLARE_TYPE(GEODE_EXPORT)
  typedef block_partition_t Base;

protected:
  partition_t(const int ranks, const sections_t& sections);
public:
  virtual ~partition_t();

  // Count lines belonging to a given rank
  virtual uint64_t rank_count_lines(const int rank) const = 0;

  // List all lines belonging to a given rank
  virtual Array<const line_t> rank_lines(const int rank) const = 0;
};

// Create an empty partition, typically for sentinel use
GEODE_EXPORT Ref<const partition_t> empty_partition(const int ranks, const int slice);

// Test self consistency of a partition
void partition_test(const partition_t& partition);

}
}
