// Generate the stored positions and exact values for tie_test.cc
//
// usage: bazel run -c opt //pentago/cascade:gen_tie_cases 2>/dev/null | grep -v "^(peak"
//
// Prints C++ initializers for three tables of {raw board, value for the player to move}:
//   sound_cases: random and sensible-play boards at every slice 18-35 (claims must be sound)
//   exact_cases: late boards (28-35 stones) whose verdict needs free replies (claims must be exact)
//   edge_cases: late boards where some move's rotation ties or hands the opponent the win (exact)
#include "pentago/cascade/tie.h"
#include "pentago/base/score.h"
#include "pentago/mid/midengine.h"
#include "pentago/utility/random.h"
#include "pentago/utility/wall_time.h"
#include <cstdio>
#include <vector>
using namespace pentago;
using std::vector;

static int exact(const high_board_t board) {
  static const Array<halfsuper_s> workspace = midsolve_workspace(18);
  for (const auto& [raw, v] : midsolve(board, workspace))
    if (raw == board.raw()) return v;
  GEODE_ASSERT(false);
}

static high_board_t random_position(Random& random, const int stones) {
  for (;;) {
    const auto board = high_board_t::from_board(random_board(random, stones), false);
    if (!board.done()) return board;
  }
}

// Random play that never ends the game early
static high_board_t sensible_position(Random& random, const int stones) {
  for (;;) {
    side_t side[2] = {0, 0};
    int n = 0;
    for (; n < stones; n++) {
      const int t = n & 1;
      bool moved = false;
      for (int attempt = 0; attempt < 100 && !moved; attempt++) {
        const int c = random.uniform<int>(0, 36);
        const side_t x = side_t(1) << (16*(c/9) + c%9);
        if (x & (side[0] | side[1])) continue;
        const int q = random.uniform<int>(0, 4), d = random.uniform<int>(0, 2) ? 1 : -1;
        const auto after = high_board_t(t ? side[0] : side[0] | x, t ? side[1] | x : side[1], 2*n + 1).rotate(q, d);
        if (won(side[t] | x) || won(after.side(0)) || won(after.side(1))) continue;
        side[0] = after.side(0), side[1] = after.side(1), moved = true;
      }
      if (!moved) break;
    }
    if (n == stones) return high_board_t(side[0], side[1], 2*stones);
  }
}

// Does some move of the player to move rotate into a tie, or hand the opponent the win?
static bool rotation_edge(const high_board_t board) {
  const int t = board.turn();
  for (const int c : range(64)) {
    const side_t x = side_t(1) << c;
    if (!(x & side_mask) || (x & (board.side(0) | board.side(1)))) continue;
    const auto placed = board.place(c);
    if (won(placed.side(t))) continue;
    for (const int q : range(4))
      for (const int d : {-1, 1}) {
        const auto after = placed.rotate(q, d);
        const bool wt = won(after.side(t)), wo = won(after.side(!t));
        if (wo) return true;  // Tie or opponent win
        (void)wt;
      }
  }
  return false;
}

static void print(const char* name, const vector<high_board_t>& boards) {
  printf("const tie_case_t %s[] = {\n", name);
  for (size_t i = 0; i < boards.size(); i++)
    printf("%s{%lluull,%d},%s", i % 3 ? "" : "  ", (unsigned long long)boards[i].raw(), exact(boards[i]),
           i % 3 == 2 || i + 1 == boards.size() ? "\n" : " ");
  printf("};\n\n");
}

int main() {
  Random random(31415);
  tie_prover_t prover;

  // Soundness: unfiltered boards, plus boards where the prover claims a tie, plus boards where it
  // claims one side holds but the value isn't a tie (so the other claim must fail)
  vector<high_board_t> sound;
  for (int stones = 18; stones <= 35; stones++) {
    const int plain = stones < 24 ? 4 : 10, want = stones < 22 ? 2 : stones < 24 ? 3 : 6;
    for (const int i : range(plain))
      sound.push_back(i & 1 ? sensible_position(random, stones) : random_position(random, stones));
    int ties = 0, halves = 0;
    for (int attempt = 0; attempt < 3000 && (ties < want || halves < want); attempt++) {
      const auto board = attempt & 1 ? sensible_position(random, stones) : random_position(random, stones);
      const auto r = prover.prove(board, 3, 0.2);
      if (r.tie() && ties < want) sound.push_back(board), ties++;
      else if (!r.tie() && (r.mover >= 0 || r.other >= 0) && halves < want && exact(board)) sound.push_back(board), halves++;
    }
    fprintf(stderr, "sound %d: ties %d, half claims on non-ties %d\n", stones, ties, halves);
  }

  // Late boards where full-depth proofs are fast, preferring verdicts that need free replies
  vector<high_board_t> exact_boards, edge;
  for (int stones = 28; stones <= 35; stones++) {
    int deep = 0, shallow = 0, edges = 0, losses = 0, deep_wins = 0;
    for (int attempt = 0; attempt < 50000 && (deep < 20 || shallow < 5 || edges < 15 || losses < 5); attempt++) {
      const auto board = attempt & 1 ? sensible_position(random, stones) : random_position(random, stones);
      const auto start = wall_time();
      const auto r = prover.prove(board, 36 - stones, 0.3);
      if (r.timeout || (wall_time() - start).seconds() > 0.2) continue;
      if (rotation_edge(board) && edges < 15 && stones >= 30) {
        edge.push_back(board), edges++;
        continue;
      }
      const bool is_deep = r.mover > 0 || r.other > 0;
      if (r.mover < 0 && losses < 5) exact_boards.push_back(board), losses++;  // Losses are rare late
      else if (is_deep && deep < 20 && (r.tie() || deep_wins < 8)) {
        exact_boards.push_back(board), deep++;
        deep_wins += !r.tie();
      } else if (!is_deep && shallow < 5) exact_boards.push_back(board), shallow++;
    }
    fprintf(stderr, "stones %d: deep %d (%d wins), shallow %d, losses %d, edge %d\n", stones, deep, deep_wins,
            shallow, losses, edges);
  }
  print("sound_cases", sound);
  print("exact_cases", exact_boards);
  print("edge_cases", edge);
  return 0;
}
