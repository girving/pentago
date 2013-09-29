// Load balancing statistics (see partition.h for actual balancing)

#include <pentago/end/load_balance.h>
#include <pentago/end/blocks.h>
#include <other/core/math/constants.h>
#include <other/core/utility/curry.h>
#include <other/core/utility/openmp.h>
#include <other/core/python/Class.h>
#include <other/core/utility/interrupts.h>
namespace pentago {
namespace end {

OTHER_DEFINE_TYPE(load_balance_t)

load_balance_t::load_balance_t() {}
load_balance_t::~load_balance_t() {}

Range<Box<int64_t>*> load_balance_t::boxes() {
  return range(&lines,&block_local_ids+1);
}

void load_balance_t::enlarge(const load_balance_t& load) {
  lines.enlarge(load.lines);
  line_blocks.enlarge(load.line_blocks);
  line_nodes.enlarge(load.line_nodes);
  blocks.enlarge(load.blocks);
  block_nodes.enlarge(load.block_nodes);
  block_local_ids.enlarge(load.block_local_ids);
}

static Ref<load_balance_t> local_load_balance(RawArray<const line_t> lines, RawArray<const local_block_t> blocks) {
  check_interrupts();
  const auto load = new_<load_balance_t>();
  for (Box<int64_t>& box : load->boxes())
    box = 0;
  load->lines = lines.size();
  for (auto& line : lines) {
    load->line_blocks += line.length;
    const auto shape = line.section.shape();
    load->line_nodes += shape[line.dimension]*block_shape(shape.remove_index(line.dimension),line.block_base).product();
  }
  load->blocks = blocks.size();
  for (auto& block : blocks) {
    load->block_nodes += block_shape(block.section.shape(),block.block).product();
    load->block_local_ids.max = max(load->block_local_ids.max,block.local_id.id);
  }
  load->block_local_ids = load->block_local_ids.max;
  return load;
}

static void flip_min(load_balance_t& load) {
  for (auto& box : load.boxes())
    box.min = -box.min;
}

Ref<const load_balance_t> load_balance(const reduction_t<int64_t,max_op>& reduce_max, RawArray<const line_t> lines, RawArray<const local_block_t> blocks) {
  const auto load = local_load_balance(lines,blocks);
  flip_min(load);
  const bool root = reduce_max(RawArray<int64_t>(12,&load->lines.min));
  if (root)
    flip_min(load);
  else
    for (auto& box : load->boxes())
      box = 0;
  return load;
}

static void serial_load_balance_helper(const partition_t& partition, const Range<int> rank_range, load_balance_t* load) {
  for (const int rank : rank_range)
    load->enlarge(local_load_balance(partition.rank_lines(rank),partition.rank_blocks(rank)));
}

Ref<const load_balance_t> serial_load_balance(const partition_t& partition) {
  const int count = min(16*thread_counts().x,partition.ranks);
  OTHER_ASSERT(count);
  vector<Ref<load_balance_t>> loads;
  for (int j=0;j<count;j++)
    loads.push_back(new_<load_balance_t>());
  for (const int j : range(count))
    threads_schedule(CPU,curry(serial_load_balance_helper,ref(partition),partition_loop(partition.ranks,count,j),&*loads[j]));
  threads_wait_all_help();
  for (const int j : range(1,count))
    loads[0]->enlarge(loads[j]);
  return loads[0];
}

string str(const load_balance_t& load) {
  #define FIELD(s,name) format(#s " = %lld %lld (%.4g)",load.name.min,load.name.max,load.name.min?double(load.name.max)/load.name.min:inf)
  return  FIELD(lines,lines)
    +'\n'+FIELD(line blocks,line_blocks)
    +'\n'+FIELD(line nodes,line_nodes)
    +'\n'+FIELD(blocks,blocks)
    +'\n'+FIELD(block nodes,block_nodes);
}

}
}
using namespace pentago::end;

void wrap_load_balance() {
  typedef load_balance_t Self;
  Class<Self>("load_balance_t")
    .OTHER_FIELD(lines)
    .OTHER_FIELD(line_blocks)
    .OTHER_FIELD(line_nodes)
    .OTHER_FIELD(blocks)
    .OTHER_FIELD(block_nodes)
    .OTHER_FIELD(block_local_ids)
    .OTHER_STR()
    ;
  OTHER_FUNCTION(serial_load_balance)
}
