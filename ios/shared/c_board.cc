// C interface to high_board_t, since Swift can't interoperate with C++.

#include "c_board.h"
#include "../pentago/high/board.h"
#include "../pentago/mid/midengine.h"
#include "../pentago/mid/internal.h"
#include "../pentago/base/gen/halfsuper_wins.h"

using namespace pentago;

static inline high_board_t t(high_board_s s) { return high_board_t(s); }

bool board_done(high_board_s b) { return t(b).done(); }
int board_immediate_value(high_board_s b) { return t(b).immediate_value(); }
high_board_s board_place_bit(high_board_s b, int bit) { return t(b).place(bit).s; }
high_board_s board_place_xy(high_board_s b, int x, int y) { return t(b).place(x, y).s; }
high_board_s board_rotate(high_board_s b, int q, int d) { return t(b).rotate(q, d).s; }

int board_workspace_size(high_board_s b) {
  return midsolve_workspace_size(t(b).count());
}

info_t make_info_t(high_board_s b, int workspace_size) {
  return make_info(t(b), workspace_size);
}

inner_t make_inner_t(const info_t I, const int n) {
  return make_inner(I, n);
}

static int set_results(high_board_value_t results[1+18+8*18], const mid_values_t& values) {
  for (const int i : range(values.size())) {
    const auto& [k, v] = values[i];
    auto& dst = results[i];
    dst.board = k.s;
    dst.value = v;
  }
  return values.size();
}

int board_midsolve_traverse(high_board_s board, const halfsupers_t* supers,
                            high_board_value_t results[1+18+8*18]) {
  mid_values_t values;
  midsolve_traverse(board, supers, values);
  return set_results(results, values);
}


int board_midsolve(const high_board_s s, high_board_value_t results[1+18+8*18]) {
  const auto board = t(s);
  const int workspace_size = midsolve_workspace_size(board.count());
  halfsupers_t* workspace = (halfsupers_t*)malloc(sizeof(halfsupers_t) * workspace_size);
  const auto values = midsolve(board, RawArray<halfsupers_t>(workspace_size, workspace));
  free(workspace);
  return set_results(results, values);
}

namespace pentago {
void die(const char* msg) {
  fprintf(stderr, "%s\n", msg);
  exit(1);
}
}
