// Exact values of boards with 18 or more stones, by a cascade of provers

#include "pentago/cascade/cascade.h"
#include "pentago/base/score.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/wasm_alloc.h"
#ifndef __wasm__
#include "pentago/utility/wall_time.h"
#endif
namespace pentago {

using std::make_tuple;
using std::max;

#ifdef __wasm__
WASM_IMPORT double now();  // Seconds, from JavaScript
static double clock() { return now(); }
#else
static double clock() { return wall_time().seconds(); }
#endif

WASM_SMALL cascade_t::cascade_t(const cascade_options_t& options)
  : options(options), engine(options.super_bits) {
  if (options.cutoffs) engine.cutoffs.prover = &tie_;
  engine.cutoffs.depth = options.cutoff_depth;
  tie_.set_clock(clock);
}

cascade_t::~cascade_t() {}

WASM_SMALL int cascade_t::superengine(const high_board_t board, const double seconds) {
  NON_WASM_ASSERT(!board.middle() && !board.done());
  // The engine wants the player to move as side 0
  return engine.search(board.side(board.turn()), board.side(!board.turn()), seconds, clock);
}

WASM_SMALL void cascade_t::midsolve(const high_board_t board, mid_values_t& results) {
  NON_WASM_ASSERT(board.count() >= 18);
  if (!workspace.size()) {
#ifdef __wasm__
    workspace = wasm_buffer<halfsuper_s>(midsolve_workspace_size(18));
#else
    workspace_owner = midsolve_workspace(18);
    workspace = workspace_owner;
#endif
  }
  results.clear();
#ifdef __wasm__
  // midsolve allocates scratch space, and the wasm allocator never frees, so rewind it afterwards
  const auto mark = wasm_heap_mark();
#endif
  const auto supers = midsolve_internal(board, workspace);
  midsolve_traverse(board, supers.data(), results);
#ifdef __wasm__
  wasm_heap_rewind(mark);
#endif
}

WASM_SMALL int cascade_t::midsolve(const high_board_t board) {
  NON_WASM_ASSERT(!board.middle());
  mid_values_t values;
  midsolve(board, values);
  for (const auto& [raw, v] : values)
    if (raw == board.raw()) return v;
  NON_WASM_ASSERT(false, "midsolve did not return the root");
  return 0;
}

WASM_SMALL cascade_result_t cascade_t::value(const high_board_t board) {
  NON_WASM_ASSERT(!board.middle() && !board.done() && board.count() >= 18);
  const int s = superengine(board, options.super_seconds);
  if (s != 2) return {s, cascade_stage_t::superengine};
  if (tie_.prove(board, options.tie_depth, options.tie_seconds).tie()) return {0, cascade_stage_t::tie};
  return {midsolve(board), cascade_stage_t::midsolve};
}

// The cheap stages for each rotation of a placed board, until seconds after start.  Returns whether
// every rotation is known.
WASM_SMALL bool cascade_t::cheap_placement(const high_board_t placed, cascade_result_t results[8], const double start,
                                const double seconds) {
  NON_WASM_ASSERT(placed.middle() && !placed.done() && placed.count() >= 18);
  bool known = true;
  for (int s = 0; s < 8; s++) {
    auto& r = results[s];
    const auto child = placed.rotate(s >> 1, s & 1 ? -1 : 1);
    r = {2, cascade_stage_t::superengine};
    if (child.done()) {
      r.value = child.immediate_value();
    } else if (clock() - start < seconds) {
      r.value = superengine(child, options.super_seconds);
      if (r.value == 2 && tie_.prove(child, options.tie_depth, options.tie_seconds).tie())
        r = {0, cascade_stage_t::tie};
    }
    known &= r.value != 2;
  }
  return known;
}

// Settle the unknown rotations of a placed board with one midsolve
WASM_SMALL void cascade_t::finish_placement(const high_board_t placed, cascade_result_t results[8]) {
  mid_values_t values;
  midsolve(placed, values);
  for (int s = 0; s < 8; s++) {
    auto& r = results[s];
    if (r.value != 2) continue;
    const auto child = placed.rotate(s >> 1, s & 1 ? -1 : 1).raw();
    for (const auto& [raw, v] : values)
      if (raw == child) r = {v, cascade_stage_t::midsolve};
    NON_WASM_ASSERT(r.value != 2);
  }
}

WASM_SMALL void cascade_t::placement(const high_board_t placed, cascade_result_t results[8]) {
  if (!cheap_placement(placed, results, clock(), options.placement_seconds))
    finish_placement(placed, results);
}

WASM_SMALL bool cascade_t::children(const high_board_t board, mid_values_t& results) {
  NON_WASM_ASSERT(!board.middle() && !board.done() && board.count() >= 17);
  results.clear();
  const double start = clock();

  // Cheap stages for every placement, giving up once too many placements are unknown
  const auto empty = board.empty_mask();
  cascade_result_t rs[36][8];
  int unknown[36], n = 0, unknowns = 0;
  for (int bit = 0; bit < 64; bit++) {
    if (!(empty >> bit & 1)) continue;
    const auto placed = board.place(bit);
    if (!placed.done() && !cheap_placement(placed, rs[n], start, options.children_seconds)) {
      if (unknowns == options.max_dense || clock() - start >= options.children_seconds) return false;
      unknown[unknowns++] = bit;
    }
    n++;
  }
  for (int i = 0; i < unknowns; i++)
    finish_placement(board.place(unknown[i]), rs[popcount(empty & ((side_t(1) << unknown[i]) - 1))]);

  // Assemble in midsolve_traverse's order
  int value = -1;
  n = 0;
  for (int bit = 0; bit < 64; bit++) {
    if (!(empty >> bit & 1)) continue;
    const auto placed = board.place(bit);
    int v;
    if (placed.done()) {
      v = placed.immediate_value();
    } else {
      v = -1;
      for (int s = 0; s < 8; s++) {
        results.append(make_tuple(placed.rotate(s >> 1, s & 1 ? -1 : 1).raw(), rs[n][s].value));
        v = max(v, -rs[n][s].value);
      }
    }
    results.append(make_tuple(placed.raw(), v));
    value = max(value, v);
    n++;
  }
  results.append(make_tuple(board.raw(), value));
  return true;
}

}  // namespace pentago

#ifdef __wasm__
// Exports for the web client.  The cascade lives for the life of the instance.
using namespace pentago;
inline void* operator new(size_t, void* p) noexcept { return p; }
static cascade_t* cascade = 0;

// Values of a board, its placements, and their rotations, as from midsolve, if the cheap stages
// (plus at most max_dense midsolves) settle them; returns whether they did.  super_bits sizes the
// forward search's table on first use.
WASM_EXPORT WASM_SMALL int cascade_children(const raw_t raw, const int super_bits, mid_values_t* results) {
  if (!cascade) {
    cascade_options_t options;
    options.super_bits = super_bits;
    cascade = new (malloc(sizeof(cascade_t))) cascade_t(options);
  }
  return cascade->children(high_board_t::from_raw(raw), *results);
}
#endif  // __wasm__
