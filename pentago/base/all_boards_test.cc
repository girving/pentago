#include "pentago/base/all_boards.h"
#include "pentago/base/count.h"
#include "pentago/base/hash.h"
#include "pentago/base/symmetry.h"
#include "pentago/utility/array.h"
#include "pentago/utility/hash.h"
#include "pentago/utility/range.h"
#include "pentago/utility/log.h"
#include "gtest/gtest.h"
#include <unordered_map>
#include <unordered_set>
namespace pentago {
namespace {

using std::get;
using std::make_pair;
using std::min;
using std::unordered_map;
using std::unordered_set;

void helper(const int symmetries) {
  vector<int> sizes, all_sizes;
  vector<string> hashes, all_hashes;
  if (symmetries == 1) {
    sizes = {1,36,1260,21420,353430,3769920,38955840};
    hashes = {"05fe405753166f125559e7c9ac558654f107c7e9", "6f436e8746348d7a0a2f9416323e0ac9f7ee8992",
              "5ffad230681a476cbc33901b8ed7d473e721b995", "5e6eda2c60533b0eed85113e33ceb07289a7aef2",
              "15e8125d1798b6e18f5d02691c5a5123ce72f3b2", "499d9155a02f8a07e5c7314dd5dcc1561cd059d6",
              "f392633216cbeaeb22e7073f2d953805509b168f"};
  } else if (symmetries == 8) {
    sizes = {1,6,165,2715,44481,471870,4871510,36527160,264802788};
    hashes = {"05fe405753166f125559e7c9ac558654f107c7e9", "2f3618ae0973fd076255bf892cb1594470b203ef",
              "a59b31ec91dc175aab1bfd94fb8e4ff5a8fc1fe0", "1d72107a01b55d13b13e509cf700f31ff2d62990",
              "5322f3629de31817a217666ac4bc197577b06ca3", "208ca65e8cb26c691f6ec14bf398d1dbec92ccb0",
              "7b79248adf142680e1ae37a54ebbeb3e30491458", "fe422248b4f41a49f8dbc884f49c39eb253fb2c6",
              "af40db6c2575017dc62f9837f1055a9cb12e87c2"};
  } else if (symmetries == 2048) {
    sizes = {1,3,30,227,2013,13065,90641,493844,2746022,12420352,56322888};
    hashes = {"05fe405753166f125559e7c9ac558654f107c7e9", "597241c65bdbdde28269031b3035bfff95c8dfa4",
              "bedc958370b06c3951a3a55010801ab1b53a36af", "09dfa654eaaccfca1c50d08ee21d23ccdbe737f2",
              "c1982d1252ae8209b9bd4fa7ffed05708a72e648", "764e7486ff3c1955eb5967c896e7cdb3778c0dd3",
              "60ddd31f7c7b3e5817c07c952eebb8e350efc6c2", "75e740e79eab6f43a3f22c081c37916333246ae7",
              "cba975b696a4b2a0f6dbb0ad23507009a47af2c9", "345f651e20f0ac9e2a11872f0d574858d53f7963",
              "60cb22beee25e8a3d1ac548de6247bbba70d7298"};
    all_sizes = {1,3,36,286,2816,15772,105628,565020,3251984};
    all_hashes = {"05fe405753166f125559e7c9ac558654f107c7e9", "597241c65bdbdde28269031b3035bfff95c8dfa4",
                  "a6def1687498ba310f0f33dda4c3de1b845ba8ae", "2464973e1ef7bde09ba5eef9fe65731955371590",
                  "1c301e5953ae69d37781d46b9fe104f9c8371491", "212f49c5171495c2ff93fd0ca71d2c27e31ede45",
                  "709cf5ada8411202a55616b68ca40649c9bdbfab", "8275ac68db3dde43e25c90da19e57d3ec932f907",
                  "9b32620990c2a784b9fdbb0c6c89caa51b68b5b4"};
  }
  ASSERT_EQ(sizes.size(), hashes.size());
  ASSERT_EQ(all_sizes.size(), all_hashes.size());
  for (const int n : range(symmetries == 1 ? 5 : 6)) {
    const auto boards = all_boards(n, symmetries);
    if (0 && n < 3) {
      slog("\n\n-------------------------------- symmetries %d, n %d, count %d "
           "---------------------------------\n", symmetries, n, boards.size());
      for (const auto b : boards)
        slog(str_board(b));
    }
    const auto h = portable_hash(boards);
    slog("n = %d, count = %d, hash = %s", n, boards.size(), h);
    ASSERT_EQ(sizes[n], boards.size());
    ASSERT_EQ(hashes[n], h);
    if (symmetries == 2048 && n < all_sizes.size()) {
      std::sort(boards.begin(), boards.end());
      ASSERT_EQ(boards.size(), count_boards(n, 2048));
      const unordered_set<board_t> boards_set(boards.begin(), boards.end());
      ASSERT_EQ(boards_set.size(), boards.size());
      const auto approx = all_boards_list(n);
      const auto h = portable_hash(approx);
      slog("approx: count = %d, hash = %s, ratio %g",
           approx.size(), h, approx.size() / sizes[n]);
      ASSERT_LE(sizes[n], approx.size());
      ASSERT_LE(approx.size(), all_sizes[n]);
      ASSERT_EQ(all_hashes[n], h);
      for (auto& board : approx)
        board = get<0>(superstandardize(board));
      const unordered_set<board_t> approx_set(approx.begin(), approx.end());
      for (const auto b : boards)
        ASSERT_TRUE(contains(approx_set, b));
    }
  }
}

TEST(all_boards, raw) { helper(1); }
TEST(all_boards, all) { helper(8); }
TEST(all_boards, super) { helper(2048); }

// Verify that boards with at most 2 stones all map to different hashes mod 511.
// This ensures that such positions will never disappear from the transposition
// table as the result of a collision.
TEST(all_boards, small_hashes) {
  vector<board_t> boards;
  for (const int n : {0, 1, 2})
    extend(boards, all_boards(n, 2048));
  unordered_set<uint16_t> hash8, hash9;
  for (const auto board : boards) {
    const auto h = hash_board(board);
    hash8.insert(h & ((1<<8)-1));
    hash9.insert(h & ((1<<9)-1));
  }
  ASSERT_LT(hash8.size(), boards.size());
  ASSERT_EQ(hash9.size(), boards.size());
}

void sample_test(const int n, const int steps) {
  Array<section_t> sections = all_boards_sections(n);

  // Sort buckets in preparation for binary search
  unordered_map<Vector<uint8_t,2>,Array<quadrant_t>> sorted;
  for (uint8_t b=0;b<=9;b++)
    for (uint8_t w=0;w<=9-b;w++) {
      auto bucket = get<0>(rotation_minimal_quadrants(b,w)).copy();
      std::sort(bucket.begin(), bucket.end());
      sorted.insert(make_pair(vec(b,w), bucket));
    }

  // Generate a bunch of random boards, and check that each one occurs in a section
  Random random(175131);
  for (const int step __attribute__((unused)) : range(steps)) {
    const board_t board = random_board(random,n);
    const section_t s = count(board);
    const auto [ss, g] = s.standardize<8>();
    // Does this section exist?
    const int si = int(std::lower_bound(sections.begin(),sections.end(),ss)-sections.begin());
    if (!(sections.valid(si) && sections[si]==ss)) {
      slog("missing section %s, standard %s, n %d", s.counts, ss.counts, n);
      ASSERT_EQ(get<0>(ss.standardize<8>()), ss);
      GEODE_ASSERT(false);
    }
    // Does the board occur in the section?
    const board_t sboard = transform_board(symmetry_t(g,0),board);
    for (int i=0;i<4;i++) {
      RawArray<const quadrant_t> bucket = check_get(sorted, ss.counts[i]);
      const quadrant_t q = quadrant(sboard,i);
      ASSERT_TRUE(std::binary_search(
          bucket.begin(), bucket.end(), get<0>(rotation_standardize_quadrant(q))));
    }
  }
}

TEST(all_boards, sample) {
  for (const int n : range(37)) {
    const auto steps = min(uint64_t(100000), count_boards(n, 2048));
    slog("sample test: n = %d, steps = %d", n, steps);
    sample_test(n, steps);
  }
}

TEST(all_boards, rmin) {
  // Check inverse
  for (quadrant_t q=0;q<quadrant_count;q++) {
    const auto standard = rotation_standardize_quadrant(q);
    const int ir = rotation_minimal_quadrants_inverse[q];
    const auto rmin = get<0>(rotation_minimal_quadrants(count(q)));
    ASSERT_TRUE(rmin.valid(ir/4));
    ASSERT_EQ(rmin[ir/4], get<0>(standard));
    ASSERT_EQ(transform_board(symmetry_t(0,ir&3),get<0>(standard)), q);
  }

  // Check that all quadrants changed by reflection occur in pairs in the first part of the array
  for (uint8_t b=0;b<=9;b++) {
    for (uint8_t w=0;w<=9-b;w++) {
      const auto rmin_moved = rotation_minimal_quadrants(vec(b,w));
      const auto rmin = get<0>(rmin_moved);
      const int moved = get<1>(rmin_moved);
      ASSERT_TRUE((moved&1)==0 && (!moved || moved<rmin.size()));
      for (int i=0;i<rmin.size();i++) {
        const quadrant_t q = rmin[i];
        const quadrant_t qr = pack(reflections[unpack(q,0)],reflections[unpack(q,1)]);
        const int ir = rotation_minimal_quadrants_inverse[qr]/4;
        ASSERT_EQ(i^(i<moved), ir);
      }
    }
  }
}

}  // namespace
}  // namespace pentago
