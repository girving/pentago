// Partitioning of sections and lines for MPI purposes

#include "pentago/end/simple_partition.h"
#include "pentago/base/all_boards.h"
#include "pentago/end/blocks.h"
#include "pentago/end/config.h"
#include "pentago/utility/ceil_div.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/index.h"
#include "pentago/utility/large.h"
#include "pentago/utility/log.h"
#include "pentago/utility/memory.h"
#include "pentago/utility/sort.h"
#include "pentago/utility/random.h"
#include "pentago/utility/const_cast.h"
#include "pentago/utility/box.h"
#include "pentago/utility/sqr.h"
#include "pentago/utility/str.h"
namespace pentago {
namespace end {

using std::min;
using std::max;
using std::make_tuple;

static const uint64_t worst_line = 8*8*8*420;

// A regular grid of 1D block lines with the same section, dimension, and shape
struct chunk_t {
  // Section information
  section_t section;
  Vector<int,4> shape;

  // Line information
  unsigned dimension : 2; // Line dimension
  uint8_t length; // Line length in blocks
  Box<Vector<uint8_t,3>> blocks; // Ranges of blocks along the other three dimensions
  int count; // Number of lines in this range = blocks.volume()
  int node_step; // Number of nodes in all blocks except possibly those at the end of lines
  int line_size; // All lines in this line range have the same size

  // Running total information, meaningful for owners only
  uint64_t block_id; // Unique id of first block (others are indexed consecutively)
  uint64_t node_offset; // Total number of nodes in all blocks in lines before this
};
static_assert(sizeof(chunk_t)==64,"");

simple_partition_t::simple_partition_t(const int ranks, const shared_ptr<const sections_t>& sections,
                                       bool save_work)
  : partition_t(ranks, sections)
  , owner_excess(numeric_limits<double>::infinity())
  , total_excess(numeric_limits<double>::infinity())
  , max_rank_blocks(0), max_rank_nodes(0) {
  GEODE_ASSERT(ranks>0);
  Scope scope("simple partition");
  thread_time_t time(partition_kind,unevent);

  // Each section is a 4D array of blocks, and computing a section requires iterating
  // over all its 1D block lines along the four different dimensions.  One of the
  // dimensions (the most expensive one based on branching factor) is chosen as the
  // owner: the rank which computes an owning line owns all contained blocks.

  // For now, our partitioning strategy is completely unaware of both graph topology
  // and network topology.  We simply lay out all lines end to end and give each
  // process a contiguous chunk of lines.  This is done independently for owning
  // and non-owning lines.  Since the number of total lines is quite large, we take
  // advantage of the fact that for each section dimension, there are at most 8
  // different line sizes (depending on whether the line is partial along the 3
  // other dimensions).  This optimization makes the partitioning computation cheap
  // enough to do redundantly across all processes.

  // Construct the sets of lines
  vector<chunk_t> owner_lines, other_lines;
  uint64_t owner_line_count = 0, other_line_count = 0;
  for (auto section : sections->sections) {
    const auto shape = section.shape();
    const Vector<uint8_t,4> blocks_lo(shape/block_size),
                            blocks_hi(ceil_div(shape,block_size));
    const auto sums = section.sums();
    GEODE_ASSERT(sums.sum()<36);
    const int owner_k = sums.argmin();
    const int old_owners = owner_lines.size();
    for (int k=0;k<4;k++) {
      if (section.counts[k].sum()<9) {
        const auto shape_k = shape.remove_index(k);
        const auto blocks_lo_k = blocks_lo.remove_index(k),
                   blocks_hi_k = blocks_hi.remove_index(k);
        vector<chunk_t>& lines = owner_k==k ? owner_lines : other_lines;
        uint64_t& line_count = owner_k==k ? owner_line_count : other_line_count;
        for (int i0=0;i0<2;i0++) {
          for (int i1=0;i1<2;i1++) {
            for (int i2=0;i2<2;i2++) {
              const auto I = vec(i0,i1,i2);
              chunk_t chunk;
              for (int a=0;a<3;a++) {
                if (I[a]) {
                  chunk.blocks.min[a] = blocks_lo_k[a];
                  chunk.blocks.max[a] = blocks_hi_k[a];
                } else {
                  chunk.blocks.min[a] = 0;
                  chunk.blocks.max[a] = blocks_lo_k[a];
                }
              }
              chunk.count = Box<Vector<int,3>>(chunk.blocks).volume();
              if (chunk.count) {
                chunk.section = section;
                chunk.shape = shape;
                chunk.dimension = k;
                chunk.length = blocks_hi[k];
                const int cross_section = block_shape(shape_k,chunk.blocks.min).product();
                chunk.line_size = shape[k]*cross_section;
                chunk.node_step = block_size*cross_section;
                chunk.block_id = chunk.node_offset = (uint64_t)1<<60; // Garbage value
                lines.push_back(chunk);
                line_count += chunk.count;
              }
            }
          }
        }
      }
    }
    // Record the first owner line corresponding to this section
    GEODE_ASSERT(owner_lines.size() > size_t(old_owners));
    const_cast_(first_owner_line)[section] = old_owners;
  }
  // Fill in offset information for owner lines
  uint64_t block_id = 0, node_offset = 0;
  for (auto& chunk : owner_lines) {
    chunk.block_id = block_id;
    chunk.node_offset = node_offset;
    block_id += chunk.length*chunk.count;
    node_offset += (uint64_t)chunk.line_size*chunk.count;
  }
  if (verbose() && sections->sections.size())
    slog("total lines = %d %d, total blocks = %s, total nodes = %s",
         owner_line_count, other_line_count, large(block_id), large(node_offset));
  // Remember
  const_cast_(this->owner_lines) = asarray(owner_lines).copy();
  const_cast_(this->other_lines) = asarray(other_lines).copy();
  GEODE_ASSERT(sections->total_blocks==block_id);
  GEODE_ASSERT(sections->total_nodes==node_offset);

  // Partition lines between processes, attempting to equalize (1) owned work and (2) total work.
  Array<uint64_t> work_nodes(ranks), work_penalties(ranks);

  const_cast_(owner_starts) = partition_lines(work_nodes,work_penalties,owner_lines,CHECK_CAST_INT(owner_line_count));
  if (verbose() && sections->sections.size()) {
    const auto sum = work_nodes.sum(), max = work_nodes.max();
    const_cast_(owner_excess) = (double)max/sum*ranks;
    const auto penalty_excess = (double)work_penalties.max()/work_penalties.sum()*ranks;
    slog("slice %s owned work: all = %s, range = %s %s, excess = %g (%g)",
         sections->slice, large(sum), large(work_nodes.min()), large(max), owner_excess, penalty_excess);
    GEODE_ASSERT(max<=(owner_line_count+ranks-1)/ranks*worst_line);
  }
  if (save_work)
    const_cast_(owner_work) = work_nodes.copy();

  const_cast_(other_starts) = partition_lines(work_nodes,work_penalties,other_lines,CHECK_CAST_INT(other_line_count));
  if (verbose() && sections->sections.size()) {
    auto sum = work_nodes.sum(), max = work_nodes.max();
    const_cast_(total_excess) = (double)max/sum*ranks;
    const auto penalty_excess = (double)work_penalties.max()/work_penalties.sum()*ranks;
    slog("slice %s total work: all = %s, range = %s %s, excess = %g (%g)",
         sections->slice, large(sum), large(work_nodes.min()), large(max), total_excess,
         penalty_excess);
    GEODE_ASSERT(max<=(owner_line_count+other_line_count+ranks-1)/ranks*worst_line);
  }
  if (save_work)
    const_cast_(other_work) = work_nodes;

  // Compute the maximum number of blocks owned by a rank
  uint64_t max_blocks = 0,
           max_nodes = 0;
  Vector<uint64_t,2> start;
  for (int rank=0;rank<ranks;rank++) {
    const auto end = rank_offsets(rank+1);
    max_blocks = max(max_blocks,end[0]-start[0]);
    max_nodes = max(max_nodes,end[1]-start[1]);
    start = end;
  }
  const_cast_(this->max_rank_blocks) = CHECK_CAST_INT(max_blocks);
  const_cast_(this->max_rank_nodes) = max_nodes;
}

simple_partition_t::~simple_partition_t() {}

// Compute a penalty amount for a line.  This is based on (1) the number of blocks and (2) the total memory required by the blocks.
// Penalizing lines based purely on memory is suboptimal, since it results in a wildly varying number of blocks/lines assigned to
// different ranks.  Thus, we artificially inflate small blocks and very short lines in order to even things out.
static inline uint64_t line_penalty(const chunk_t& chunk) {
  const int block_penalty = sqr(sqr(block_size))*2/3;
  return max(chunk.line_size,block_penalty*max((int)chunk.length,4));
}

// Can the remaining work fit within the given bound?
template<bool record> bool simple_partition_t::fit(RawArray<uint64_t> work_nodes, RawArray<uint64_t> work_penalties, RawArray<const chunk_t> lines, const uint64_t bound, RawArray<Vector<int,2>> starts) {
  Vector<int,2> start;
  int p = 0;
  if (start[0]==lines.size()) {
    if (record)
      starts[p] = start;
    goto success;
  }
  for (;p<work_penalties.size();p++) {
    if (record)
      starts[p] = start;
    uint64_t free = bound-work_penalties[p];
    for (;;) {
      const auto& chunk = lines[start[0]];
      const uint64_t penalty = line_penalty(chunk);
      if (free < penalty)
        break;
      const int count = min(chunk.count-start[1], CHECK_CAST_INT(free/penalty));
      free -= count*penalty;
      if (record) {
        work_nodes[p] += count*(uint64_t)chunk.line_size;
        work_penalties[p] += count*penalty;
        GEODE_ASSERT(work_penalties[p]<=bound);
      }
      start[1] += count;
      if (start[1] == chunk.count) {
        start = vec(start[0]+1,0);
        if (start[0]==lines.size())
          goto success;
      }
    }
  }
  return false; // Didn't manage to distribute all lines
  success:
  if (record)
    starts.slice(p+1, starts.size()).fill(start);
  return true;
}

// Divide a set of lines between processes
Array<const Vector<int,2>> simple_partition_t::partition_lines(RawArray<uint64_t> work_nodes, RawArray<uint64_t> work_penalties, RawArray<const chunk_t> lines, const int line_count) {
  // Compute a lower bound for how much work each rank needs to do
  const int ranks = work_penalties.size();
  const uint64_t sum_done = work_penalties.sum(),
                 max_done = work_penalties.max();
  uint64_t left = 0;
  for (auto& chunk : lines)
    left += chunk.count*line_penalty(chunk);
  uint64_t lo = max(max_done,(sum_done+left+ranks-1)/ranks);

  // Compute an upper bound based on assuming each line is huge, and dividing them equally between ranks
  uint64_t hi = max_done+(line_count+ranks-1)/ranks*worst_line;
  GEODE_ASSERT(fit<false>(work_nodes,work_penalties,lines,hi));

  // Binary search to find the minimum maximum amount of work
  while (lo+1 < hi) {
    uint64_t mid = (lo+hi)/2;
    (fit<false>(work_nodes,work_penalties,lines,mid) ? hi : lo) = mid;
  }

  // Compute starts and update work
  Array<Vector<int,2>> starts(ranks+1,uninit);
  bool success = fit<true>(work_nodes,work_penalties,lines,hi,starts);
  GEODE_ASSERT(success);
  GEODE_ASSERT(starts.back()==vec(lines.size(),0));
  return starts;
}

uint64_t simple_partition_t::memory_usage() const {
  return sizeof(simple_partition_t)
       + pentago::memory_usage(sections)
       + pentago::memory_usage(owner_lines)
       + pentago::memory_usage(other_lines)
       + pentago::memory_usage(first_owner_line)
       + pentago::memory_usage(owner_starts)
       + pentago::memory_usage(other_starts)
       + pentago::memory_usage(owner_work)
       + pentago::memory_usage(other_work);
}

Vector<uint64_t,2> simple_partition_t::rank_offsets(int rank) const {
  // This is essentially rank_lines copied and pruned to touch only the first line 
  GEODE_ASSERT(0<=rank && rank<=ranks);
  if (rank==ranks)
    return vec(sections->total_blocks,sections->total_nodes);
  const auto start = owner_starts[rank];
  if (start[0]==owner_lines.size())
    return vec(sections->total_blocks,sections->total_nodes);
  const auto& chunk = owner_lines[start[0]];
  return vec(chunk.block_id+(uint64_t)chunk.length*start[1],
             chunk.node_offset+(uint64_t)chunk.line_size*start[1]);
}

Array<const line_t> simple_partition_t::rank_lines(int rank) const {
  return concat<line_t>(rank_lines(rank, true), rank_lines(rank,false));
}

Array<line_t> simple_partition_t::rank_lines(int rank, bool owned) const {
  GEODE_ASSERT(0<=rank && rank<ranks);
  RawArray<const chunk_t> all_lines = owned?owner_lines:other_lines;
  RawArray<const Vector<int,2>> starts = owned?owner_starts:other_starts;
  const auto start = starts[rank], end = starts[rank+1];
  vector<line_t> result;
  for (int r : range(start[0], min(end[0]+1, all_lines.size()))) {
    const auto& chunk = all_lines[r];
    const Vector<int,3> sizes(chunk.blocks.shape());
    for (int k : range(r==start[0]?start[1]:0,r==end[0]?end[1]:chunk.count)) {
      line_t line;
      line.section = chunk.section;
      line.dimension = chunk.dimension;
      line.length = chunk.length;
      line.block_base = chunk.blocks.min+Vector<uint8_t,3>(decompose(sizes,k));
      result.push_back(line);
    }
  }
  return asarray(result).copy();
}

Array<const local_block_t> simple_partition_t::rank_blocks(int rank) const {
  vector<local_block_t> blocks;
  for (const auto& line : rank_lines(rank,true))
    for (const int b : range(int(line.length))) {
      local_block_t block;
      block.local_id = local_id_t(blocks.size());
      block.section = line.section;
      block.block = line.block(b);
      blocks.push_back(block);
    }
  return asarray(blocks).copy();
}

tuple<int,local_id_t> simple_partition_t::find_block(const section_t section, const Vector<uint8_t,4> block) const {
  const int rank = block_to_rank(section,block);
  const local_id_t id(CHECK_CAST_INT(block_to_id(section,block)-rank_offsets(rank)[0]));
  return make_tuple(rank,id);
}

tuple<section_t,Vector<uint8_t,4>> simple_partition_t::rank_block(const int rank, const local_id_t local_id) const {
  const uint64_t block_id = rank_offsets(rank)[0]+local_id.id;
  // Binary search to find containing line
  GEODE_ASSERT(owner_lines.size());
  int lo = 0, hi = owner_lines.size()-1;
  while (lo<hi) {
    const int mid = (lo+hi+1)/2;
    if (owner_lines[mid].block_id <= block_id)
      lo = mid;
    else
      hi = mid-1;
  }
  const chunk_t& chunk = owner_lines[lo];
  // Compute block
  const int i = CHECK_CAST_INT(block_id-chunk.block_id);
  const auto shape = concat(Vector<int,3>(chunk.blocks.shape()), Vector<int,1>(chunk.length));
  GEODE_ASSERT((unsigned)i<=(unsigned)shape.product());
  const auto I = decompose(shape, i);
  const auto block = (chunk.blocks.min + Vector<uint8_t,3>(I.remove_index(3)))
                         .insert(I[3], chunk.dimension);
  return make_tuple(chunk.section,block);
}

uint64_t simple_partition_t::rank_count_lines(int rank) const {
  return rank_count_lines(rank,false)
       + rank_count_lines(rank,true);
}

uint64_t simple_partition_t::rank_count_lines(int rank, bool owned) const {
  GEODE_ASSERT(0<=rank && rank<ranks);
  RawArray<const chunk_t> all_lines = owned?owner_lines:other_lines;
  RawArray<const Vector<int,2>> starts = owned?owner_starts:other_starts;
  const auto start = starts[rank], end = starts[rank+1];
  uint64_t result = 0;
  for (int r : range(start[0],end[0]+1))
    result += (r==end[0]?end[1]:all_lines[r].count) - (r==start[0]?start[1]:0);
  return result;
}

Vector<uint64_t,2> simple_partition_t::rank_counts(int rank) const {
  return rank_offsets(rank+1)-rank_offsets(rank);
}

// Find the rank which owns a given block
int simple_partition_t::block_to_rank(section_t section, Vector<uint8_t,4> block) const {
  // Find the line range and specific line that contains this block
  const auto line = block_to_line(section,block);
  // Perform a binary search to find the right rank.
  // Invariants: owner_starts[lo] <= line < owner_starts[hi]
  int lo = 0, hi = ranks;
  while (lo+1 < hi) {
    const int mid = (lo+hi)/2;
    const auto start = owner_starts[mid];
    (line[0]<start[0] || (line[0]==start[0] && line[1]<start[1]) ? hi : lo) = mid;
  }
  return lo;
}

// Find the line that owns a given block
Vector<int,2> simple_partition_t::block_to_line(section_t section, Vector<uint8_t,4> block) const {
  int index = check_get(first_owner_line, section);
  const int owner_k = owner_lines[index].dimension;
  const auto block_k = block.remove_index(owner_k);
  do {
    const auto& blocks = owner_lines[index].blocks;
    for (int a=0;a<3;a++)
      if (!(blocks.min[a]<=block_k[a] && block_k[a]<blocks.max[a]))
        goto skip;
    // We've locate the range of lines that contains (and owns) the block.  Now isolate the exact line. 
    {
      const Vector<int,3> I(block_k - blocks.min),
                          sizes(blocks.shape());
      return vec(index,pentago::index(sizes,I));
    }
    // The block isn't contained in this line range
    skip:;
    index++;
  } while (owner_lines.valid(index) && owner_lines[index].section==section);
  die("block_to_line failed: section %s, blocks %s, block %s",str(section),str(section_blocks(section)),str(Vector<int,4>(block)));
}

uint64_t simple_partition_t::block_to_id(section_t section, Vector<uint8_t,4> block) const {
  const auto I = block_to_line(section,block);
  const auto& chunk = owner_lines[I[0]];
  const int j = block[chunk.dimension];
  return chunk.block_id+(uint64_t)chunk.length*I[1]+j;
}

}
}
