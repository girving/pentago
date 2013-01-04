// Random partitioning of lines and blocks across processes

#include <pentago/mpi/random_partition.h>
#include <pentago/utility/index.h>
#include <pentago/utility/memory.h>
#include <other/core/python/Class.h>
#include <other/core/random/permute.h>
#include <other/core/random/counter.h>
#include <other/core/utility/const_cast.h>
#include <other/core/utility/Log.h>
#include <other/core/utility/openmp.h>
#include <other/core/utility/str.h>
namespace pentago {
namespace mpi {

using Log::cout;
using std::endl;
using namespace other;

OTHER_DEFINE_TYPE(random_partition_t)

// All lines in a given section pointing in the same direction
struct sheaf_t {
  section_t section;
  uint8_t dimension : 2;
  uint8_t length : 6;
  Vector<uint8_t,3> shape; // section_blocks(section).remove_index(dimension)
  int lines; // shape.product()
  int line_offset;
};
BOOST_STATIC_ASSERT(sizeof(sheaf_t)==20);

random_partition_t::random_partition_t(const uint128_t key, const int ranks, const sections_t& sections)
  : partition_t(ranks,sections)
  , key(key)
  , total_lines(0) {

  OTHER_ASSERT(ranks>0 && sections.slice<36);
  Log::Scope scope("random partition");
  thread_time_t time(partition_kind,unevent);

  // Assign contiguous indices to all lines
  {
    Array<sheaf_t> sheafs;
    int64_t line_offset = 0;
    for (const auto& section : sections.sections) {
      const auto shape = section_blocks(section);
      for (const uint8_t dimension : range(4))
        if (section.counts[dimension].sum()<9) {
          sheaf_t sheaf;
          sheaf.section = section;
          sheaf.dimension = dimension;
          sheaf.length = shape[dimension];
          sheaf.shape = Vector<uint8_t,3>(shape.remove_index(dimension));
          sheaf.lines = Vector<int,3>(sheaf.shape).product();
          sheaf.line_offset = line_offset;
          const_cast_(sheaf_id).set(tuple(section,dimension),sheafs.append(sheaf));
          line_offset += sheaf.lines;
        }
    }
    const_cast_(total_lines) = line_offset;
    OTHER_ASSERT(total_lines==line_offset);
    const_cast_(this->sheafs) = sheafs; 
  }
}

random_partition_t::~random_partition_t() {}

uint64_t random_partition_t::memory_usage() const {
  return sizeof(random_partition_t)
       + pentago::memory_usage(sections)
       + pentago::memory_usage(sheafs)
       + pentago::memory_usage(sheaf_id);
}

uint64_t random_partition_t::rank_count_lines(const int rank) const {
  return partition_loop(total_lines,ranks,rank).size();
}

Array<const line_t> random_partition_t::rank_lines(const int rank) const {
  const auto range = partition_loop(total_lines,ranks,rank);
  Array<line_t> lines(range.size(),false);
  for (const int n : range)
    lines[n-range.lo] = nth_line(n);
  return lines;
}

Array<const local_block_t> random_partition_t::rank_blocks(const int rank) const {
  Array<local_block_t> blocks;
  const auto line_range = partition_loop(total_lines,ranks,rank);
  for (const int n : line_range) {
    const line_t line = nth_line(n);
    for (const int b : range(int(line.length))) {
      const auto block = line.block(b);
      if (owner_dimension(line.section,block)==line.dimension) {
        local_block_t local;
        local.local_id = local_id_t((n-line_range.lo)<<6|b);
        local.section = line.section;
        local.block = block;
        blocks.append(local);
      }
    }
  }
  return blocks;
}

Vector<uint64_t,2> random_partition_t::rank_counts(const int rank) const {
  uint64_t nodes = 0;
  const auto blocks = rank_blocks(rank);
  for (const auto& block : blocks)
    nodes += block_shape(block.section.shape(),block.block).product();
  return vec(uint64_t(blocks.size()),nodes);
}

Tuple<int,local_id_t> random_partition_t::find_block(const section_t section, const Vector<uint8_t,4> block) const {
  // Compute details about owning line
  const auto dimension = owner_dimension(section,block);
  const int* sheaf_n = sheaf_id.get_pointer(tuple(section,dimension));
  if (!sheaf_n)
    die("random_partition_t::find_block: section %s, block %d,%d,%d,%d not part of partition with slice %d",str(section),block[0],block[1],block[2],block[3],sections->slice);
  const auto& sheaf = sheafs[*sheaf_n];
  const int line_n = random_unpermute(total_lines,key,sheaf.line_offset+index(Vector<int,3>(sheaf.shape),Vector<int,3>(block.remove_index(dimension))));
  // Compute rank and local block id
  const int rank = partition_loop_inverse(total_lines,ranks,line_n),
            local_line_n = line_n - partition_loop(total_lines,ranks,rank).lo;
  const local_id_t local_block_id(local_line_n<<6|block[dimension]);
  return tuple(rank,local_block_id);
}

Tuple<section_t,Vector<uint8_t,4>> random_partition_t::rank_block(const int rank, const local_id_t local_id) const {
  const int local_line_n = local_id.id>>6;
  const auto range = partition_loop(total_lines,ranks,rank);
  OTHER_ASSERT((unsigned)local_line_n<(unsigned)range.size());
  const int line_n = range.lo+local_line_n;
  const auto line = nth_line(line_n);
  const int b = local_id.id&63;
  OTHER_ASSERT(b<line.length);
  return tuple(line.section,line.block(b));
}

line_t random_partition_t::nth_line(const int n) const {
  // Map n through our random permutation
  const int pn = random_permute(total_lines,key,n);
  // Do a binary search to find correct sheaf
  int lo = 0, hi = sheafs.size()-1;
  while (lo<hi) {
    const int mid = (lo+hi+1)>>1;
    if (sheafs[mid].line_offset <= pn)
      lo = mid;
    else
      hi = mid-1;
  }
  const auto& sheaf = sheafs[lo];
  const int index = pn-sheaf.line_offset;
  assert(range(sheaf.lines).contains(index));
  // Construct result line
  line_t line;
  line.section = sheaf.section;
  line.dimension = sheaf.dimension;
  line.length = sheaf.length;
  line.block_base = Vector<uint8_t,3>(decompose(Vector<int,3>(sheaf.shape),index));
  return line;
}

uint8_t random_partition_t::owner_dimension(const section_t section, const Vector<uint8_t,4> block) const {
  // Count how many lines we intersect
  int count = 0;
  Vector<uint8_t,4> dimensions;
  for (const int i : range(4))
    if (section.counts[i].sum()<9)
      dimensions[count++] = i;
  OTHER_ASSERT(count);
  // Randomly choose one of the lines as primary
  return dimensions[count==1 ? 0 : threefry(key,block_event(section,block)) % count];
}

}
}
using namespace pentago::mpi;

void wrap_random_partition() {
  typedef random_partition_t Self;
  Class<Self>("random_partition_t")
    .OTHER_INIT(uint128_t,int,const sections_t&)
    ;
}
