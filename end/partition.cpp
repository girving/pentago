// Abstract base class for partitioning of lines and blocks

#include <pentago/end/partition.h>
#include <pentago/end/history.h>
#include <pentago/end/simple_partition.h>
#include <pentago/end/blocks.h>
#include <pentago/end/config.h>
#include <pentago/utility/ceil_div.h>
#include <pentago/utility/debug.h>
#include <other/core/math/constants.h>
#include <other/core/utility/curry.h>
#include <other/core/utility/Log.h>
#include <other/core/utility/openmp.h>
#include <other/core/python/Class.h>
namespace pentago {
namespace end {

using Log::cout;
using std::endl;
using namespace other;

OTHER_DEFINE_TYPE(partition_t)

partition_t::partition_t(const int ranks, const sections_t& sections)
  : ranks(ranks)
  , sections(ref(sections)) {}

partition_t::~partition_t() {}

Ref<const partition_t> empty_partition(const int ranks, const int slice) {
  return new_<simple_partition_t>(ranks,new_<sections_t>(slice,Array<section_t>()));
}

void partition_test(const partition_t& partition) {
  // Extract all information from partition
  Hashtable<event_t> all_lines;
  Hashtable<Tuple<section_t,Vector<uint8_t,4>>,Tuple<int,local_id_t>> block_info;
  for (const int rank : range(partition.ranks)) {
    const auto lines = partition.rank_lines(rank);
    OTHER_ASSERT(partition.rank_count_lines(rank)==uint64_t(lines.size()));
    for (const auto& line : lines) {
      OTHER_ASSERT(line.length==ceil_div(line.section.shape()[line.dimension],block_size));
      const bool unique = all_lines.set(line_event(line.section,line.dimension,line.block_base));
      OTHER_ASSERT(unique);
    }
    const auto counts = partition.rank_counts(rank);
    const auto blocks = partition.rank_blocks(rank);
    OTHER_ASSERT(counts.x==uint64_t(blocks.size()));
    uint64_t nodes = 0;
    for (const auto& block : blocks) {
      const auto info = tuple(rank,block.local_id);
      OTHER_ASSERT(partition.find_block(block.section,block.block)==info);
      OTHER_ASSERT(partition.rank_block(info.x,info.y)==tuple(block.section,block.block));
      const bool unique = block_info.set(tuple(block.section,block.block),info);
      OTHER_ASSERT(unique);
      nodes += block_shape(block.section.shape(),block.block).product();
    }
    OTHER_ASSERT(counts.y==nodes);
  }
  // Compare against single rank simple partition
  const auto simple = new_<simple_partition_t>(1,partition.sections);
  const auto correct_lines = simple->rank_lines(0);
  for (const auto& line : correct_lines) {
    const auto event = line_event(line.section,line.dimension,line.block_base);
    if (!all_lines.contains(event))
      THROW(RuntimeError,"missing line: %s",str_event(event));
  }
  OTHER_ASSERT(correct_lines.size()==all_lines.size());
  for (const auto& info : block_info)
    simple->find_block(info.key.x,info.key.y); // Verify that the block is supposed to exist
  for (const auto& block : simple->rank_blocks(0))
    OTHER_ASSERT(block_info.contains(tuple(block.section,block.block)));
}

}
}
using namespace pentago::end;

void wrap_partition() {
  typedef partition_t Self;
  Class<Self>("partition_t")
    .OTHER_FIELD(ranks)
    .OTHER_FIELD(sections)
    ;
  OTHER_FUNCTION(partition_test)
}
