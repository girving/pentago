// Partitioning of sections and lines for MPI purposes

#include <pentago/mpi/partition.h>
#include <pentago/mpi/utility.h>
#include <pentago/all_boards.h>
#include <pentago/utility/ceil_div.h>
#include <pentago/utility/index.h>
#include <pentago/utility/large.h>
#include <pentago/utility/memory.h>
#include <other/core/array/sort.h>
#include <other/core/python/Class.h>
#include <other/core/python/stl.h>
#include <other/core/random/Random.h>
#include <other/core/math/constants.h>
#include <other/core/utility/const_cast.h>
#include <other/core/vector/Interval.h>
#include <other/core/utility/Log.h>
#include <other/core/utility/str.h>
namespace pentago {
namespace mpi {

using Log::cout;
using std::endl;

vector<Array<const section_t>> descendent_sections(const section_t root, const int max_slice) {
  Log::Scope scope("dependents");
  OTHER_ASSERT(0<=max_slice && max_slice<=35);
  OTHER_ASSERT(root.sum()<=max_slice);

  // Recursively compute all sections that root depends on
  vector<Array<section_t>> slices(max_slice+1);
  Hashtable<section_t> seen;
  Array<section_t> stack;
  stack.append(root);
  while (stack.size()) {
    section_t section = stack.pop().standardize<8>().x;
    if (seen.set(section)) {
      int n = section.sum();
      slices.at(n).append(section);
      if (n < max_slice)
        for (int i=0;i<4;i++)
          if (section.counts[i].sum()<9)
            stack.append(section.child(i));
    }
  }

  // Sort each slice
  for (auto& slice : slices)
    sort(slice);

  // Make const
  return vector<Array<const section_t>>(slices.begin(),slices.end());
}

OTHER_DEFINE_TYPE(partition_t)

static const uint64_t worst_line = 8*8*8*420;

// A regular grid of 1D block lines with the same section, dimension, and shape
struct chunk_t {
  // Section information
  section_t section;
  Vector<int,4> shape;

  // Line information
  int dimension; // Line dimension
  int length; // Line length in blocks
  Box<Vector<int,3>> blocks; // Ranges of blocks along the other three dimensions
  int count; // Number of lines in this range = blocks.volume()
  int node_step; // Number of nodes in all blocks except possibly those at the end of lines
  int line_size; // All lines in this line range have the same size

  // Running total information, meaningful for owners only
  uint64_t block_id; // Unique id of first block (others are indexed consecutively)
  uint64_t node_offset; // Total number of nodes in all blocks in lines before this
};
BOOST_STATIC_ASSERT(sizeof(chunk_t)==88);

partition_t::partition_t(const int ranks, const int slice, Array<const section_t> sections, bool save_work)
  : ranks(ranks), slice(slice), sections(sections)
  , owner_excess(inf), total_excess(inf)
  , total_blocks(0), total_nodes(0), max_rank_blocks(0), max_rank_nodes(0) {
  OTHER_ASSERT(ranks>0);
  Log::Scope scope("partition");
  thread_time_t time(partition_kind);

  // Verify that caller didn't lie about slice
  for (const auto& section : sections)
    OTHER_ASSERT(section.sum()==slice);

  // Invert sections
  for (int s=0;s<sections.size();s++) 
    const_cast_(section_id).set(sections[s],s);

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
  Array<chunk_t> owner_lines, other_lines;
  uint64_t owner_line_count = 0, other_line_count = 0;
  for (auto section : sections) {
    const auto shape = section.shape(),
               blocks_lo = shape/block_size,
               blocks_hi = ceil_div(shape,block_size);
    const auto sums = section.sums();
    OTHER_ASSERT(sums.sum()<36);
    const int owner_k = sums.argmin();
    const int old_owners = owner_lines.size();
    for (int k=0;k<4;k++) {
      if (section.counts[k].sum()<9) {
        const auto shape_k = shape.remove_index(k),
                   blocks_lo_k = blocks_lo.remove_index(k),
                   blocks_hi_k = blocks_hi.remove_index(k);
        Array<chunk_t>& lines = owner_k==k?owner_lines:other_lines;
        uint64_t& line_count = owner_k==k?owner_line_count:other_line_count;
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
              chunk.count = chunk.blocks.volume();
              if (chunk.count) {
                chunk.section = section;
                chunk.shape = shape;
                chunk.dimension = k;
                chunk.length = blocks_hi[k];
                const int cross_section = block_shape(shape_k,chunk.blocks.min).product();
                chunk.line_size = shape[k]*cross_section;
                chunk.node_step = block_size*cross_section;
                chunk.block_id = chunk.node_offset = (uint64_t)1<<60; // Garbage value
                lines.append(chunk);
                line_count += chunk.count;
              }
            }
          }
        }
      }
    }
    // Record the first owner line corresponding to this section
    OTHER_ASSERT(owner_lines.size()>old_owners);
    const_cast_(first_owner_line).set(section,old_owners);
  }
  // Fill in offset information for owner lines
  uint64_t block_id = 0, node_offset = 0;
  for (auto& chunk : owner_lines) {
    chunk.block_id = block_id;
    chunk.node_offset = node_offset;
    block_id += chunk.length*chunk.count;
    node_offset += (uint64_t)chunk.line_size*chunk.count;
  }
  if (verbose() && sections.size())
    cout << "total lines = "<<owner_line_count<<' '<<other_line_count<<", total blocks = "<<large(block_id)<<", total nodes = "<<large(node_offset)<<endl;
  // Remember
  const_cast_(this->owner_lines) = owner_lines;
  const_cast_(this->other_lines) = other_lines;
  const_cast_(this->total_blocks) = block_id;
  const_cast_(this->total_nodes) = node_offset;

  // Partition lines between processes, attempting to equalize (1) owned work and (2) total work.
  Array<uint64_t> work_nodes(ranks), work_penalties(ranks);

  const_cast_(owner_starts) = partition_lines(work_nodes,work_penalties,owner_lines,owner_line_count);
  if (verbose() && sections.size()) {
    const auto sum = work_nodes.sum(), max = work_nodes.max();
    const_cast_(owner_excess) = (double)max/sum*ranks;
    const auto penalty_excess = (double)work_penalties.max()/work_penalties.sum()*ranks;
    cout << "slice "<<slice<<" owned work: all = "<<large(sum)<<", range = "<<large(work_nodes.min())<<' '<<large(max)<<", excess = "<<owner_excess<<" ("<<penalty_excess<<')'<<endl;
    OTHER_ASSERT(max<=(owner_line_count+ranks-1)/ranks*worst_line);
  }
  if (save_work)
    const_cast_(owner_work) = work_nodes.copy();

  const_cast_(other_starts) = partition_lines(work_nodes,work_penalties,other_lines,other_line_count);
  if (verbose() && sections.size()) {
    auto sum = work_nodes.sum(), max = work_nodes.max();
    const_cast_(total_excess) = (double)max/sum*ranks;
    const auto penalty_excess = (double)work_penalties.max()/work_penalties.sum()*ranks;
    cout << "slice "<<slice<<" total work: all = "<<large(sum)<<", range = "<<large(work_nodes.min())<<' '<<large(max)<<", excess = "<<total_excess<<" ("<<penalty_excess<<')'<<endl;
    OTHER_ASSERT(max<=(owner_line_count+other_line_count+ranks-1)/ranks*worst_line);
  }
  if (save_work)
    const_cast_(other_work) = work_nodes;

  // Compute the maximum number of blocks owned by a rank
  uint64_t max_blocks = 0,
           max_nodes = 0;
  Vector<uint64_t,2> start;
  for (int rank=0;rank<ranks;rank++) {
    const auto end = rank_offsets(rank+1);
    max_blocks = max(max_blocks,end.x-start.x);
    max_nodes = max(max_nodes,end.y-start.y);
    start = end;
  }
  OTHER_ASSERT(max_blocks<(1<<30));
  const_cast_(this->max_rank_blocks) = max_blocks;
  const_cast_(this->max_rank_nodes) = max_nodes;
}

partition_t::~partition_t() {}

// Compute a penalty amount for a line.  This is based on (1) the number of blocks and (2) the total memory required by the blocks.
// Penalizing lines based purely on memory is suboptimal, since it results in a wildly varying number of blocks/lines assigned to
// different ranks.  Thus, we artificially inflate small blocks and very short lines in order to even things out.
static inline uint64_t line_penalty(const chunk_t& chunk) {
  const int block_penalty = sqr(sqr(block_size))*2/3;
  return max(chunk.line_size,block_penalty*max(chunk.length,4));
}

// Can the remaining work fit within the given bound?
template<bool record> bool partition_t::fit(RawArray<uint64_t> work_nodes, RawArray<uint64_t> work_penalties, RawArray<const chunk_t> lines, const uint64_t bound, RawArray<Vector<int,2>> starts) {
  Vector<int,2> start;
  int p = 0;
  if (start.x==lines.size()) {
    if (record)
      starts[p] = start;
    goto success;
  }
  for (;p<work_penalties.size();p++) {
    if (record)
      starts[p] = start;
    uint64_t free = bound-work_penalties[p];
    for (;;) {
      const auto& chunk = lines[start.x];
      const uint64_t penalty = line_penalty(chunk);
      if (free < penalty)
        break;
      const int count = min(chunk.count-start.y,free/penalty);
      free -= count*penalty;
      if (record) {
        work_nodes[p] += count*(uint64_t)chunk.line_size;
        work_penalties[p] += count*penalty;
        OTHER_ASSERT(work_penalties[p]<=bound);
      }
      start.y += count;
      if (start.y == chunk.count) {
        start = vec(start.x+1,0);
        if (start.x==lines.size())
          goto success;
      }
    }
  }
  return false; // Didn't manage to distribute all lines
  success:
  if (record)
    starts.slice(p+1,starts.size()).fill(start);
  return true;
}

// Divide a set of lines between processes
Array<const Vector<int,2>> partition_t::partition_lines(RawArray<uint64_t> work_nodes, RawArray<uint64_t> work_penalties, RawArray<const chunk_t> lines, const int line_count) {
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
  OTHER_ASSERT(fit<false>(work_nodes,work_penalties,lines,hi));

  // Binary search to find the minimum maximum amount of work
  while (lo+1 < hi) {
    uint64_t mid = (lo+hi)/2;
    (fit<false>(work_nodes,work_penalties,lines,mid) ? hi : lo) = mid;
  }

  // Compute starts and update work
  Array<Vector<int,2>> starts(ranks+1,false);
  bool success = fit<true>(work_nodes,work_penalties,lines,hi,starts);
  OTHER_ASSERT(success);
  OTHER_ASSERT(starts.last()==vec(lines.size(),0));
  return starts;
}

uint64_t partition_t::memory_usage() const {
  return sizeof(partition_t)
       + pentago::memory_usage(sections)
       + pentago::memory_usage(owner_lines)
       + pentago::memory_usage(other_lines)
       + pentago::memory_usage(first_owner_line)
       + pentago::memory_usage(owner_starts)
       + pentago::memory_usage(other_starts)
       + pentago::memory_usage(owner_work)
       + pentago::memory_usage(other_work);
}

Vector<uint64_t,2> partition_t::rank_offsets(int rank) const {
  // This is essentially rank_lines copied and pruned to touch only the first line 
  OTHER_ASSERT(0<=rank && rank<=ranks);
  if (rank==ranks)
    return vec(total_blocks,total_nodes);
  const auto start = owner_starts[rank];
  if (start.x==owner_lines.size())
    return vec(total_blocks,total_nodes);
  const auto& chunk = owner_lines[start.x];
  return vec(chunk.block_id+chunk.length*start.y,
             chunk.node_offset+(uint64_t)chunk.line_size*start.y);
}

Array<line_t> partition_t::rank_lines(int rank, bool owned) const {
  OTHER_ASSERT(0<=rank && rank<ranks);
  RawArray<const chunk_t> all_lines = owned?owner_lines:other_lines;
  RawArray<const Vector<int,2>> starts = owned?owner_starts:other_starts;
  const auto start = starts[rank], end = starts[rank+1];
  Array<line_t> result;
  for (int r : range(start.x,min(end.x+1,all_lines.size()))) {
    const auto& chunk = all_lines[r];
    const auto sizes = chunk.blocks.sizes();
    for (int k : range(r==start.x?start.y:0,r==end.x?end.y:chunk.count)) {
      line_t line;
      line.section = chunk.section;
      line.dimension = chunk.dimension;
      line.length = chunk.length;
      line.node_step = chunk.node_step;
      line.block_base = chunk.blocks.min+decompose(sizes,k);
      line.block_id = chunk.block_id+chunk.length*k;
      line.node_offset = chunk.node_offset+(uint64_t)chunk.line_size*k;
      result.append(line);
    }
  }
  return result;
}

uint64_t partition_t::rank_count_lines(int rank, bool owned) const {
  OTHER_ASSERT(0<=rank && rank<ranks);
  RawArray<const chunk_t> all_lines = owned?owner_lines:other_lines;
  RawArray<const Vector<int,2>> starts = owned?owner_starts:other_starts;
  const auto start = starts[rank], end = starts[rank+1];
  uint64_t result = 0;
  for (int r : range(start.x,end.x+1))
    result += (r==end.x?end.y:all_lines[r].count) - (r==start.x?start.y:0);
  return result;
}

// Find the rank which owns a given block
int partition_t::block_to_rank(section_t section, Vector<int,4> block) const {
  // Find the line range and specific line that contains this block
  const auto line = block_to_line(section,block);
  // Perform a binary search to find the right rank.
  // Invariants: owner_starts[lo] <= line < owner_starts[hi]
  int lo = 0, hi = ranks;
  while (lo+1 < hi) {
    const int mid = (lo+hi)/2;
    const auto start = owner_starts[mid];
    (line.x<start.x || (line.x==start.x && line.y<start.y) ? hi : lo) = mid;
  }
  return lo;
}

// Find the line that owns a given block
Vector<int,2> partition_t::block_to_line(section_t section, Vector<int,4> block) const {
  int index = first_owner_line.get(section);
  const int owner_k = owner_lines[index].dimension;
  const auto block_k = block.remove_index(owner_k);
  do {
    const auto& blocks = owner_lines[index].blocks;
    for (int a=0;a<3;a++)
      if (!(blocks.min[a]<=block_k[a] && block_k[a]<blocks.max[a]))
        goto skip;
    // We've locate the range of lines that contains (and owns) the block.  Now isolate the exact line. 
    {
      const auto I = block_k - blocks.min,
                 sizes = blocks.sizes();
      const int line = (I.x*sizes.y+I.y)*sizes.z+I.z;
      return vec(index,line);
    }
    // The block isn't contained in this line range
    skip:;
    index++;
  } while (owner_lines.valid(index) && owner_lines[index].section==section);
  die(format("block_to_line failed: section %s, blocks %s, block %s",str(section),str(section_blocks(section)),str(block)));
}

// (block_id, global node offset) for a given block
Vector<uint64_t,2> partition_t::block_offsets(section_t section, Vector<int,4> block) const {
  const auto I = block_to_line(section,block);
  const auto& chunk = owner_lines[I.x];
  const int j = block[chunk.dimension];
  return vec(chunk.block_id+chunk.length*I.y+j,
             chunk.node_offset+(uint64_t)chunk.line_size*I.y+chunk.node_step*j);
}

static void partition_test() {
  const int stones = 24;
  const uint64_t total = 1921672470396,
                 total_blocks = 500235319;

  // This number is slightly lower than the number from 'analyze approx' because we skip lines with no moves.
  const uint64_t total_lines = 95263785;

  // Grab all 24 stone sections
  Array<section_t> sections = all_boards_sections(stones,8);

  // Partition with a variety of different ranks, from 6 to 768k
  uint64_t other = -1;
  auto random = new_<Random>(877411);
  for (int ranks=3<<2;ranks<=(3<<18);ranks<<=2) {
    Log::Scope scope(format("ranks %d",ranks));
    auto partition = new_<partition_t>(ranks,stones,sections,true);

    // Check totals
    OTHER_ASSERT(partition->total_nodes==total);
    OTHER_ASSERT(partition->total_blocks==total_blocks);
    OTHER_ASSERT(partition->rank_offsets(ranks)==vec(total_blocks,total));
    OTHER_ASSERT(partition->owner_work.sum()==total);
    auto o = partition->other_work.sum();
    if (other==(uint64_t)-1)
      other = o;
    OTHER_ASSERT(other==o);
    const Interval<double> excess(1,1.06);
    OTHER_ASSERT(excess.contains(partition->owner_excess));
    OTHER_ASSERT(excess.contains(partition->total_excess));

    // Check that random blocks all occur in the partition
    for (auto section : sections) {
      const auto blocks = ceil_div(section.shape(),block_size);
      for (int i=0;i<10;i++) {
        const auto block = random->uniform(Vector<int,4>(),blocks);
        int rank = partition->block_to_rank(section,block);
        OTHER_ASSERT(0<=rank && rank<ranks);
      }
    }

    // Check max_rank_blocks
    uint64_t max_blocks = 0,
             max_lines = 0,
             rank_total_lines = 0;
    for (int rank=0;rank<ranks;rank++) {
      max_blocks = max(max_blocks,partition->rank_offsets(rank+1).x-partition->rank_offsets(rank).x);
      const auto lines = partition->rank_count_lines(rank,true)+partition->rank_count_lines(rank,false);
      max_lines = max(max_lines,lines);
      rank_total_lines += lines;
    }
    OTHER_ASSERT(max_blocks==(uint64_t)partition->max_rank_blocks);
    OTHER_ASSERT(total_lines==rank_total_lines);
    cout << "average blocks = "<<(double)total_blocks/ranks<<", max blocks = "<<max_blocks<<endl;
    cout << "average lines = "<<(double)total_lines/ranks<<", max lines = "<<max_lines<<endl;

    // For several ranks, check that the lists of lines are consistent
    if (ranks>=196608) {
      int cross = 0, total = 0;
      for (int i=0;i<100;i++) {
        const int rank = random->uniform<int>(0,ranks);
        // We should own all blocks in lines we own
        Hashtable<Tuple<section_t,Vector<int,4>>> blocks;
        const auto owned = partition->rank_lines(rank,true);
        OTHER_ASSERT((uint64_t)owned.size()==partition->rank_count_lines(rank,true));
        const auto first_offsets = partition->rank_offsets(rank),
                   last_offsets = partition->rank_offsets(rank+1);
        bool first = true;
        auto next_offset = first_offsets;
        for (const auto& line : owned)
          for (int j=0;j<line.length;j++) {
            const auto block = line.block(j);
            OTHER_ASSERT(partition->block_to_rank(line.section,block)==rank);
            const auto offsets = partition->block_offsets(line.section,block);
            if (first) {
              OTHER_ASSERT(offsets==first_offsets);
              first = false;
            }
            OTHER_ASSERT(offsets==line.block_offsets(j));
            OTHER_ASSERT(offsets.all_less(last_offsets));
            blocks.set(tuple(line.section,block));
            OTHER_ASSERT(next_offset==offsets);
            next_offset.x++;
            next_offset.y += block_shape(line.section.shape(),block).product();
          }
        OTHER_ASSERT(next_offset==last_offsets);
        // We only own some of the blocks in lines we don't own
        auto other = partition->rank_lines(rank,false);
        OTHER_ASSERT((uint64_t)other.size()==partition->rank_count_lines(rank,false));
        for (const auto& line : other)
          for (int j=0;j<line.length;j++) {
            const auto block = line.block(j);
            const bool own = partition->block_to_rank(line.section,block)==rank;
            OTHER_ASSERT(own==blocks.contains(tuple(line.section,block)));
            cross += own;
            total++;
          }
      }
      OTHER_ASSERT(total);
      cout << "cross ratio = "<<cross<<'/'<<total<<" = "<<(double)cross/total<<endl;
      OTHER_ASSERT(cross || ranks==786432);
    }
  }

  double ratio = (double)other/total;
  cout << "other/total = "<<ratio<<endl;
  OTHER_ASSERT(3.9<ratio && ratio<4);
}

}
}
using namespace other;
using namespace pentago;
using namespace pentago::mpi;

void wrap_partition() {
  OTHER_FUNCTION(partition_test)
  OTHER_FUNCTION(descendent_sections)

  typedef partition_t Self;
  Class<Self>("partition_t")
    .OTHER_INIT(int,int,Array<const section_t>,bool)
    .OTHER_FIELD(max_rank_blocks)
    .OTHER_FIELD(max_rank_nodes)
    .OTHER_FIELD(owner_starts)
    .OTHER_FIELD(other_starts)
    .OTHER_FIELD(total_nodes)
    ;
}
