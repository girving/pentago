// Helper functions for testing purposes

#include "pentago/end/block_store.h"
namespace pentago {
namespace end {

shared_ptr<accumulating_block_store_t> meaningless_block_store(
    const shared_ptr<const block_partition_t>& partition, const int rank, const int samples_per_section,
    const shared_ptr<compacting_store_t>& store);

Vector<uint64_t,3> meaningless_counts(RawArray<const board_t> boards);

void compare_blocks_with_sparse_samples(const readable_block_store_t& blocks,
                                        RawArray<const board_t> boards,
                                        RawArray<const Vector<super_t,2>> data);

void compare_blocks_with_supertensors(const readable_block_store_t& blocks,
                                      const vector<shared_ptr<supertensor_reader_t>>& readers);

tuple<Vector<uint64_t,3>,int> compare_readers_and_samples(
    const supertensor_reader_t& reader, const shared_ptr<const supertensor_reader_t> old_reader,
    RawArray<const board_t> sample_boards, RawArray<const Vector<super_t,2>> sample_wins);

}
}
