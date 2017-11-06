// Load balancing statistics (see partition.h for actual balancing)

#include "pentago/end/load_balance.h"
#include "pentago/end/blocks.h"
#include "pentago/utility/curry.h"
#include "pentago/utility/log.h"
namespace pentago {
namespace end {

using std::make_shared;
using std::min;
using std::max;

load_balance_t::load_balance_t() {}
load_balance_t::~load_balance_t() {}

Range<Box<int64_t>*> load_balance_t::boxes() {
  return range(&lines, &block_local_ids+1);
}

void load_balance_t::enlarge(const load_balance_t& load) {
  lines.enlarge(load.lines);
  line_blocks.enlarge(load.line_blocks);
  line_nodes.enlarge(load.line_nodes);
  blocks.enlarge(load.blocks);
  block_nodes.enlarge(load.block_nodes);
  block_local_ids.enlarge(load.block_local_ids);
}

static shared_ptr<load_balance_t>
local_load_balance(RawArray<const line_t> lines, RawArray<const local_block_t> blocks) {
  const auto load = make_shared<load_balance_t>();
  for (Box<int64_t>& box : load->boxes())
    box = 0;
  load->lines = lines.size();
  for (auto& line : lines) {
    load->line_blocks += line.length;
    const auto shape = line.section.shape();
    load->line_nodes += shape[line.dimension] *
                        block_shape(shape.remove_index(line.dimension), line.block_base).product();
  }
  load->blocks = blocks.size();
  for (auto& block : blocks) {
    load->block_nodes += block_shape(block.section.shape(),block.block).product();
    load->block_local_ids.max = max(load->block_local_ids.max, int64_t(block.local_id.id));
  }
  load->block_local_ids = load->block_local_ids.max;
  return load;
}

static void flip_min(load_balance_t& load) {
  for (auto& box : load.boxes())
    box.min = -box.min;
}

shared_ptr<const load_balance_t> load_balance(
    const reduction_t<int64_t,max_op>& reduce_max, RawArray<const line_t> lines,
    RawArray<const local_block_t> blocks) {
  const auto load = local_load_balance(lines,blocks);
  flip_min(*load);
  const bool root = reduce_max(RawArray<int64_t>(12,&load->lines.min));
  if (root)
    flip_min(*load);
  else
    for (auto& box : load->boxes())
      box = 0;
  return load;
}

static void serial_load_balance_helper(const partition_t* partition, const Range<int> rank_range,
                                       load_balance_t* load) {
  for (const int rank : rank_range)
    load->enlarge(*local_load_balance(partition->rank_lines(rank), partition->rank_blocks(rank)));
}

shared_ptr<const load_balance_t> serial_load_balance(const shared_ptr<const partition_t>& partition) {
  const int count = min(16*thread_counts()[0], partition->ranks);
  GEODE_ASSERT(count);
  vector<shared_ptr<load_balance_t>> loads;
  for (int j=0;j<count;j++)
    loads.push_back(make_shared<load_balance_t>());
  for (const int j : range(count))
    threads_schedule(CPU, curry(serial_load_balance_helper, partition.get(),
                                partition_loop(partition->ranks, count, j), loads[j].get()));
  threads_wait_all_help();
  for (const int j : range(1,count))
    loads[0]->enlarge(*loads[j]);
  return loads[0];
}

void load_balance_t::print() const {
  #define FIELD(s, name) \
    slog(#s " = %d %d (%.4g)", name.min, name.max, \
         name.min ? double(name.max)/name.min : numeric_limits<double>::infinity());
  FIELD(lines, lines)
  FIELD(line blocks, line_blocks)
  FIELD(line nodes, line_nodes)
  FIELD(blocks, blocks)
  FIELD(block nodes, block_nodes)
}

}
}
