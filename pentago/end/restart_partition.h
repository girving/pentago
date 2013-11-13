// Restart code for mpi computations
#pragma once

#include <pentago/end/partition.h>
#include <pentago/data/supertensor.h>
#include <geode/array/Nested.h>
namespace pentago {
namespace end {

// Partition based on existing supertensors
struct restart_partition_t : public block_partition_t {
  GEODE_DECLARE_TYPE(GEODE_EXPORT)
  typedef block_partition_t Base;
  typedef const RawArray<const Ref<const supertensor_reader_t>> Tensors;
  typedef Tuple<int,Vector<uint8_t,4>> Block;

  const Nested<const Block> partition; // lists of (tensor,block) pairs
  const Hashtable<Tuple<section_t,Vector<uint8_t,4>>,Tuple<int,local_id_t>> inv_partition;

private:
  GEODE_EXPORT restart_partition_t(const int ranks, Tensors tensors);
  GEODE_EXPORT restart_partition_t(const int ranks, const sections_t& sections, Nested<const Tuple<int,Vector<uint8_t,4>>> partition);
public:
  ~restart_partition_t();

  // Define partition_t interface
  uint64_t memory_usage() const;
  Array<const local_block_t> rank_blocks(const int rank) const override;
  Vector<uint64_t,2> rank_counts(const int rank) const override; // Expensive, since it calls rank_blocks
  Tuple<int,local_id_t> find_block(const section_t section, const Vector<uint8_t,4> block) const override;
  Tuple<section_t,Vector<uint8_t,4>> rank_block(const int rank, const local_id_t local_id) const override;

private:
  // Assign blocks to ranks contiguously, staying under the specified memory limit.
  // On failure, returns an empty array.
  static Nested<const Block> partition_blocks(const int ranks, Tensors tensors, const uint64_t memory_limit);

  // Find the smallest memory_limit s.t. partition_blocks works
  static uint64_t minimum_memory_limit(const int ranks, Tensors tensors);
};

}
}
