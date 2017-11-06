// Abstract base class for partitioning of lines and blocks

#include "pentago/end/partition.h"
#include "pentago/end/history.h"
#include "pentago/end/simple_partition.h"
#include "pentago/end/blocks.h"
#include "pentago/end/config.h"
#include "pentago/utility/ceil_div.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/curry.h"
namespace pentago {
namespace end {

using std::make_shared;

block_partition_t::block_partition_t(const int ranks, const shared_ptr<const sections_t>& sections)
  : ranks(ranks)
  , sections(sections) {}

block_partition_t::~block_partition_t() {}

partition_t::partition_t(const int ranks, const shared_ptr<const sections_t>& sections)
  : Base(ranks, sections) {}

partition_t::~partition_t() {}

shared_ptr<const partition_t> empty_partition(const int ranks, const int slice) {
  return make_shared<simple_partition_t>(ranks, make_shared<sections_t>(slice,Array<section_t>()));
}

namespace {
struct null_line_partition_t : public partition_t {
  const shared_ptr<const block_partition_t> p;

  null_line_partition_t(const shared_ptr<const block_partition_t>& p)
    : partition_t(p->ranks, p->sections)
    , p(p) {}

  uint64_t memory_usage() const {
    return p->memory_usage();
  }

  Array<const local_block_t> rank_blocks(const int rank) const {
    return p->rank_blocks(rank);
  }

  Vector<uint64_t,2> rank_counts(const int rank) const {
    return p->rank_counts(rank);
  }

  tuple<int,local_id_t> find_block(const section_t section, const Vector<uint8_t,4> block) const {
    return p->find_block(section,block);
  }

  tuple<section_t,Vector<uint8_t,4>> rank_block(const int rank, const local_id_t local_id) const {
    return p->rank_block(rank,local_id);
  }

  uint64_t rank_count_lines(const int rank) const {
    return 0;
  }

  Array<const line_t> rank_lines(const int rank) const {
    return Array<const line_t>();
  }
};
}

shared_ptr<const partition_t> null_line_partition(
    const shared_ptr<const block_partition_t>& partition) {
  return make_shared<null_line_partition_t>(partition);
}

}
}
