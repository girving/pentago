// The cascade's production use: children() of real 17 stone positions, checked against the tiled solver.
// Which stage settles a placement, and whether children() gives up, depends on timing, so every case
// asserts only that the cascade either gives up or is exactly right.

#include "pentago/cascade/cascade.h"
#include "pentago/mid/tiled.h"
#include "pentago/utility/test_assert.h"
#include "gtest/gtest.h"
namespace pentago {
namespace {

using std::unordered_map;

// Real 17 stone positions from user traffic: two the cascade settles in well under a second, one that
// takes it several seconds, and one it gives up on by default (8 placements need a midsolve)
const raw_t boards[] = {5061495148003858120ull, 2978024040404418614ull, 141300439400251708ull,
                        108387709992836170ull};

unordered_map<raw_t,int> exact(const high_board_t board) {
  tiled_options_t opts;
  opts.threads = 4;
  unordered_map<raw_t,int> values;
  for (const auto& [b, v] : midsolve_tiled(board, opts))
    values.emplace(b, v);
  return values;
}

// Returns whether children() completed, checking its values if so
bool check(cascade_t& cascade, const high_board_t board, const unordered_map<raw_t,int>& expected) {
  mid_values_t values;
  if (!cascade.children(board, values))
    return false;
  GEODE_ASSERT(values.size() == 1 + 19 + 19*8);
  for (const auto& [b, v] : values) {
    const auto it = expected.find(b);
    GEODE_ASSERT(it != expected.end() && it->second == v, tfm::format("board %d, child %d", board.raw(), b));
  }
  return true;
}

TEST(children17, matches_tiled) {
  cascade_t fast;  // Default options, as in the web client
  cascade_options_t options;
  options.max_dense = 19;
  options.children_seconds = 1e9;
  cascade_t dense(options);  // Never gives up: any placement the cheap stages miss gets midsolved
  options.super_seconds = options.tie_seconds = 0.001;
  options.children_seconds = 0.01;
  options.max_dense = 1;
  cascade_t hurried(options);  // As if on a much slower machine
  int settled = 0;
  for (const raw_t raw : boards) {
    const auto board = high_board_t::from_raw(raw);
    SCOPED_TRACE(board.name());
    PENTAGO_ASSERT_EQ(board.count(), 17);
    const auto expected = exact(board);
    settled += check(fast, board, expected);
    ASSERT_TRUE(check(dense, board, expected));
    check(hurried, board, expected);
  }
  PENTAGO_ASSERT_GE(settled, 2);  // Not vacuous: the client's options settle the easy boards
}

}  // namespace
}  // namespace pentago
