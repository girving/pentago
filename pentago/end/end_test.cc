#include "pentago/base/all_boards.h"
#include "pentago/base/count.h"
#include "pentago/end/blocks.h"
#include "pentago/end/check.h"
#include "pentago/end/config.h"
#include "pentago/end/fast_compress.h"
#include "pentago/end/partition.h"
#include "pentago/end/predict.h"
#include "pentago/end/random_partition.h"
#include "pentago/end/simple_partition.h"
#include "pentago/utility/ceil_div.h"
#include "pentago/utility/log.h"
#include "pentago/utility/memory_usage.h"
#include "pentago/utility/char_view.h"
#include "pentago/utility/curry.h"
#include "pentago/utility/arange.h"
#include "gtest/gtest.h"
#include <unordered_set>

namespace pentago {
namespace end {
namespace {

using std::get;
using std::max;
using std::min;
using std::make_pair;
using std::make_shared;
using std::make_tuple;
using std::unordered_set;

// Test self consistency of a partition
void partition_test(const partition_t& partition) {
  // Extract all information from partition
  unordered_set<event_t> all_lines;
  unordered_map<tuple<section_t,Vector<uint8_t,4>>,tuple<int,local_id_t>> block_info;
  for (const int rank : range(partition.ranks)) {
    const auto lines = partition.rank_lines(rank);
    ASSERT_EQ(partition.rank_count_lines(rank), uint64_t(lines.size()));
    for (const auto& line : lines) {
      ASSERT_EQ(line.length, ceil_div(line.section.shape()[line.dimension],block_size));
      const bool unique = all_lines.insert(
          line_event(line.section,line.dimension,line.block_base)).second;
      ASSERT_TRUE(unique);
    }
    const auto counts = partition.rank_counts(rank);
    const auto blocks = partition.rank_blocks(rank);
    ASSERT_EQ(counts[0], uint64_t(blocks.size()));
    uint64_t nodes = 0;
    for (const auto& block : blocks) {
      const auto info = make_tuple(rank, block.local_id);
      ASSERT_EQ(partition.find_block(block.section, block.block), info);
      ASSERT_EQ(partition.rank_block(get<0>(info), get<1>(info)),
                make_tuple(block.section, block.block));
      const bool unique = block_info.insert(
          make_pair(make_tuple(block.section, block.block), info)).second;
      ASSERT_TRUE(unique);
      nodes += block_shape(block.section.shape(),block.block).product();
    }
    ASSERT_EQ(counts[1], nodes);
  }
  // Compare against single rank simple partition
  const simple_partition_t simple(1, partition.sections);
  const auto correct_lines = simple.rank_lines(0);
  for (const auto& line : correct_lines) {
    const auto event = line_event(line.section,line.dimension,line.block_base);
    if (!contains(all_lines, event))
      THROW(RuntimeError,"missing line: %s",str_event(event));
  }
  ASSERT_EQ(correct_lines.size(), all_lines.size());
  for (const auto& info : block_info) {
    // Verify that the block is supposed to exist
    simple.find_block(get<0>(get<0>(info)), get<1>(get<0>(info)));
  }
  for (const auto& block : simple.rank_blocks(0))
    ASSERT_TRUE(contains(block_info, make_tuple(block.section,block.block)));
}

TEST(end, partition) {
  Random random(7);
  init_threads(-1, -1);
  vector<shared_ptr<const sections_t>> slices;
  for (const int slice : range(7)) {
    // Grab half the sections of this slice
    const auto sections = all_boards_sections(slice, 8);
    random.shuffle(sections);
    slices.emplace_back(make_shared<sections_t>(slice, sections.slice_own(0, (sections.size()+1)/2)));
  }
  typedef Vector<uint8_t,2> CV;
  for (const auto& sections : descendent_sections(vec(CV(4,4),CV(4,4),CV(4,5),CV(5,4)), 35)) {
    if (sections->sections.size()) {
      slices.push_back(sections);
    }
  }
  for (const auto& sections : slices) {
    for (const int ranks : {1,2,3,5}) {
      for (const int key : {0,1,17}) {
        Scope scope(tfm::format("partition test: slice %d, ranks %d, key %d", sections->slice, ranks, key));
        if (key) {
          partition_test(random_partition_t(key, ranks, sections));
        } else {
          partition_test(simple_partition_t(ranks, sections, false));
        }
      }
    }
  }
}

TEST(end, simple_partition) {
  init_threads(-1, -1);
  const int stones = 24;
  const uint64_t total = 1921672470396,
                 total_blocks = 500235319;

  // This number is slightly lower than the number from 'analyze approx' because we skip lines with no mo
  const uint64_t total_lines = 95263785;

  // Grab all 24 stone sections
  const auto sections = make_shared<sections_t>(stones,all_boards_sections(stones,8));

  // Partition with a variety of different ranks, from 6 to 768k
  uint64_t other = -1;
  Random random(877411);
  for (int ranks=3<<2;ranks<=(3<<18);ranks<<=2) {
    Scope scope(tfm::format("ranks %d",ranks));
    const simple_partition_t partition(ranks, sections, true);

    // Check totals
    ASSERT_EQ(sections->total_nodes, total);
    ASSERT_EQ(sections->total_blocks, total_blocks);
    ASSERT_EQ(partition.rank_offsets(ranks), vec(total_blocks,total));
    ASSERT_EQ(partition.owner_work.sum(), total);
    auto o = partition.other_work.sum();
    if (other==(uint64_t)-1)
      other = o;
    ASSERT_EQ(other, o);
    const Box<double> excess(1,1.06);
    ASSERT_TRUE(excess.contains(partition.owner_excess));
    ASSERT_TRUE(excess.contains(partition.total_excess));

    // Check that random blocks all occur in the partition
    for (auto section : sections->sections) {
      const auto blocks = ceil_div(section.shape(),block_size);
      for (int i=0;i<10;i++) {
        const auto block = random.uniform(Vector<int,4>(),blocks);
        int rank = partition.block_to_rank(section,Vector<uint8_t,4>(block));
        ASSERT_LE(0, rank);
        ASSERT_LT(rank, ranks);
      }
    }

    // Check max_rank_blocks
    uint64_t max_blocks = 0,
             max_lines = 0,
             rank_total_lines = 0;
    for (int rank=0;rank<ranks;rank++) {
      max_blocks = max(max_blocks,partition.rank_offsets(rank+1)[0]-partition.rank_offsets(rank)[0]);
      const auto lines = partition.rank_count_lines(rank,true)+partition.rank_count_lines(rank,false);
      max_lines = max(max_lines,lines);
      rank_total_lines += lines;
    }
    ASSERT_EQ(max_blocks, (uint64_t)partition.max_rank_blocks);
    ASSERT_EQ(total_lines, rank_total_lines);
    slog("average blocks = %g, max blocks = %d", (double)total_blocks/ranks, max_blocks);
    slog("average lines = %g, max lines = %d", (double)total_lines/ranks, max_lines);

    // For several ranks, check that the lists of lines are consistent
    if (ranks>=196608) {
      int cross = 0, total = 0;
      for (int i=0;i<100;i++) {
        const int rank = random.uniform<int>(0,ranks);
        // We should own all blocks in lines we own
        unordered_set<tuple<section_t,Vector<uint8_t,4>>> blocks;
        const auto owned = partition.rank_lines(rank,true);
        ASSERT_EQ((uint64_t)owned.size(), partition.rank_count_lines(rank,true));
        const auto first_offsets = partition.rank_offsets(rank),
                   last_offsets = partition.rank_offsets(rank+1);
        bool first = true;
        auto next_offset = first_offsets;
        for (const auto& line : owned)
          for (int j=0;j<line.length;j++) {
            const auto block = line.block(j);
            ASSERT_EQ(partition.block_to_rank(line.section,block), rank);
            const auto id = partition.block_to_id(line.section,block);
            if (first) {
              ASSERT_EQ(id, first_offsets[0]);
              first = false;
            }
            ASSERT_LT(id, last_offsets[0]);
            blocks.insert(make_tuple(line.section,block));
            ASSERT_EQ(next_offset[0], id);
            next_offset[0]++;
            next_offset[1] += block_shape(line.section.shape(),block).product();
          }
        ASSERT_EQ(next_offset, last_offsets);
        // We only own some of the blocks in lines we don't own
        auto other = partition.rank_lines(rank,false);
        ASSERT_EQ((uint64_t)other.size(), partition.rank_count_lines(rank,false));
        for (const auto& line : other)
          for (int j=0;j<line.length;j++) {
            const auto block = line.block(j);
            const bool own = partition.block_to_rank(line.section,block)==rank;
            ASSERT_EQ(own, (blocks.find(make_tuple(line.section,block)) != blocks.end()));
            cross += own;
            total++;
          }
      }
      ASSERT_TRUE(total);
      slog("cross ratio = %d/%d = %g", cross, total, (double)cross/total);
      ASSERT_TRUE(cross || ranks==786432);
    }
  }

  double ratio = (double)other/total;
  slog("other/total = %g", ratio);
  ASSERT_TRUE(3.9<ratio && ratio<4);
}

TEST(end, counts) {
  init_threads(-1, -1);
  for (const int slice : range(5)) {
    Scope scope(tfm::format("counting slice %d", slice));
    const auto sections = make_shared<const sections_t>(slice, all_boards_sections(slice,8));
    const auto good_counts = meaningless_counts(all_boards(slice, 1));
    uint64_t good_nodes = 0;
    for (const auto& s : sections->sections) {
      good_nodes += s.shape().product();
    }
    for (const int key : {0,1,17}) {
      Scope scope(tfm::format("partition key %d", key));
      const auto partition = key ? shared_ptr<const partition_t>(
                                       make_shared<random_partition_t>(key, 1, sections))
                                 : make_shared<simple_partition_t>(1, sections, false);
      const auto store = make_shared<compacting_store_t>(estimate_block_heap_size(*partition, 0));
      const auto blocks = meaningless_block_store(partition, 0, 0, store);
      slog("blocks = %d, correct = %d", blocks->total_nodes, good_nodes);
      ASSERT_EQ(blocks->total_nodes, good_nodes);
      const auto bad_counts = sum_section_counts(sections->sections, blocks->section_counts);
      slog("bad counts  = %s", bad_counts);
      slog("good counts = %s", good_counts);
      ASSERT_EQ(bad_counts, good_counts);
    }
  }
}

Array<const uint8_t> compress_check(RawArray<const Vector<super_t,2>> input, const bool local) {
  Array<const uint8_t> compressed;
  Array<Vector<super_t,2>> output;
  if (local) {
    compressed = local_fast_compress(input.copy(), unevent).copy();
    output = local_fast_uncompress(compressed, unevent).copy();
  } else {
    Array<uint8_t> buffer(64+7*memory_usage(input)/6, uninit);
    compressed = buffer.slice_own(0, fast_compress(input.copy(), buffer, unevent));
    output = Array<Vector<super_t,2>>(input.size());
    fast_uncompress(compressed, output, unevent);
  }
  GEODE_ASSERT(input == output);
  return compressed;
}

void test_fast_compress(const bool local) {
  init_threads(-1, -1);
  // Test a highly compressible sequence
  const Array<Vector<super_t,2>> regular(2*1873, uninit);
  char_view(regular).copy(char_view(arange<uint64_t>(64/4*1873)));
  const auto compressed = compress_check(regular, local);
  ASSERT_EQ(compressed[0], 1);
  const double ratio = compressed.size() / memory_usage(regular);
  ASSERT_LT(ratio, .314);
  // Test various random (incompressible) sequences
  Random random(18731);
  for (const int n : {0, 1, 1873}) {
    Array<Vector<super_t,2>> bad(n/8, uninit);
    for (auto& b : char_view(bad)) {
      b = random.bits<uint8_t>();
    }
    const auto compressed = compress_check(bad, local);
    ASSERT_EQ(compressed[0], 0);
    ASSERT_EQ(compressed.size(), memory_usage(bad)+1);
  }
}

TEST(end, fast_compress) {
  test_fast_compress(false);
}

TEST(end, local_fast_compress) {
  test_fast_compress(true);
}

struct thrasher_t {
  static const int jobs = 32;
  static const int chunks = jobs;
  static const int arrays = 16;
  static const int iterations = 1024;
  static const bool verbose = false;

  const shared_ptr<compacting_store_t> store;
  compacting_store_t::group_t group;
  const uint64_t limit;
  spinlock_t used_lock;
  uint64_t used;

  thrasher_t()
    : store(make_shared<compacting_store_t>(chunks*compacting_store_t::alignment+1,
                                            curry(&thrasher_t::unlocked_check,this)))
    , group(store,arrays)
    , limit(.9*store->heap_size)
    , used(0) {
    slog("heap size = %d, limit = %d", store->heap_size, limit);
    for (const int key : range(jobs))
      threads_schedule(CPU,curry(&thrasher_t::thrash,this,key));
    threads_wait_all();
  }

  static uint8_t sig(RawArray<const uint8_t> data) {
    uint8_t s = 0;
    for (auto& c : data)
      s ^= c;
    return s;
  }

  static string hex(RawArray<const uint8_t> data) {
    string s;
    for (const uint8_t c : data)
      s += tfm::format("%x%x",c&15,c>>4);
    return s;
  }

  void unlocked_check() {
    if (verbose)
      slog("check: used = %d", used);
    for (const auto& group : store->groups_for_testing())
      for (const int array : range(group.size()))
        if (group[array].size) {
          RawArray<const uint8_t> data(group[array].size, group[array].data);
          if (verbose)
            slog("  in check %d = '%s'", array, hex(data));
          ASSERT_EQ(sig(data), 7);
        }
    used = min(used, store->heap_next_for_testing());
  }

  void thrash(const int key) {
    Random random(key);
    for (int iter=0;iter<iterations;iter++) {
      // Lock a random array
      const int array = random.uniform<int>(0,arrays);
      {
        // Verify that the contents xor to 7
        compacting_store_t::lock_t alock(group,array);
        if (alock.get().size()) {
          if (verbose)
            slog("get %d = '%s'", array, hex(alock.get()));
          ASSERT_EQ(sig(alock.get()), 7);
        }
      }
      // Generate a new random array xoring to 7
      const Array<uint8_t> data(random.uniform<int>(1,2*compacting_store_t::alignment),uninit);
      {
        uint8_t s = 0;
        for (const int i : range(1,data.size()))
          s ^= (data[i] = random.bits<uint8_t>());
        data[0] = 7^s;
      }
      // If there's space, set the new array, otherwise clear the existing one
      compacting_store_t::lock_t alock(group,array);
      const int old_asize = int(compacting_store_t::align_size(alock.get().size()));
      const int diff = int(compacting_store_t::align_size(data.size())-old_asize);
      used_lock.lock();
      if (used+diff<=limit) {
        used += diff;
        used_lock.unlock();
        if (verbose)
          slog("set %d = '%s'", array, hex(data));
        alock.set(data);
      } else {
        used -= old_asize;
        used_lock.unlock();
        if (verbose)
          slog("clear %d", array);
        alock.set(Array<uint8_t>());
      }
    }
  }
};

TEST(end, compacting_store) {
  init_threads(-1, -1);
  thrasher_t();
}

}
}
}
