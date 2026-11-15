// Tie proofs via the undo strategy
//
// Static game representation: in the frame of the undo strategy, the Maker's winning sets are the 32
// five-in-a-row lines and their pullbacks through each of the 8 single quadrant rotations (156
// distinct sets, padded to 160).  For each side we keep, per set, the number of its stones in the
// set, for the frame itself (view 0) and for the frame after each single rotation (views 1-8), as
// three bit-planes over the 160 sets: count = p0 + 2 p1 + 4 p2.  Placing a stone ripple-adds the
// cell's set mask in each view, and every test the search needs is a few bitwise ops:
//
//   Maker wins now:              view 0, Maker count >= 4 (p2), no Breaker stone
//   Maker threat (one cell left after the next stone): view 0, Maker count 3, no Breaker stone
//   Breaker wins or ties at once after Maker rotation r: view r, Breaker count >= 4, no Maker stone
//
// A Maker win through rotation r is only a tie if the rotation also completes a Breaker line (view r,
// Breaker count 5 on a line).  That needs the Breaker to cover a whole set in view 0, which is rare,
// so we refine the first test per view only then.  Threats ignore this, which only helps the Maker.
//
// The Breaker also wins if the Maker's live sets satisfy the Erdős–Selfridge criterion (see quick).
// That and the immediate tests need only the frame view, so the search settles what children it can
// from frame views before building their other views or touching the memo.  Placements and replies
// are ordered by Erdős–Selfridge weight (live Maker sets through the cell, weighted 2^-missing).
// Positions are memoized exactly in 8-byte entries.
//
// The outer search over real moves memoizes, per position and role, the fewest free replies known
// to hold and the most known to fail, since holding is monotone in the number of free replies.  The
// player to move tries every first move for an immediate win or tie, settles the rest from frame
// views, and only then searches, with line counts after each rotation built once per call.
//
// This file avoids generated tables and std containers so that it compiles for freestanding wasm.

#include "pentago/cascade/tie.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/popcount.h"
#include <cstring>
#ifdef __wasm__
#include "pentago/utility/wasm_alloc.h"
#else
#include "pentago/base/score.h"
#include "pentago/utility/wall_time.h"
#include <cstdlib>
#endif
namespace pentago {

namespace {

const int nsets = 160;  // Rotated lines, padded

struct lset_t { uint64_t w[3]; };          // A subset of the 160 sets
struct planes_t { uint64_t p[3][3]; };     // Bit-sliced counts: [plane][word]
struct side_state_t { planes_t v[9]; };    // Counts for each view

// Plain data, filled in by init so that it can live in malloced memory.  We build our own packing
// and win tables, since those in pentago/base are generated and unavailable in freestanding wasm.
struct tables_t {
  uint16_t R[2][512];         // Quadrant rotations in both directions, matching high_board_t::rotate
  uint16_t pack3[512];        // Quadrant side -> radix 3 digits
  uint64_t win_parts[4][512]; // Per line, a 2-bit count of its quadrant segments that a quadrant covers
  uint64_t win_need;          // Per line, the number of quadrants it passes through
  side_t sets[nsets];         // Cells of each rotated line, in frame coordinates
  lset_t lines;               // Sets that are unrotated lines
  lset_t real;                // Sets that aren't padding
  side_t pulled[9][nsets];    // Set i of view v pulled back to the frame
  lset_t through[9][64];      // Sets of view v containing each frame cell, after it moves with the view
  uint64_t key_delta[2][64];  // pack(m, b) increments for a Maker or Breaker stone
  side_t rot_cell[8][64];     // Each cell after each rotation

  side_t rot(const side_t s, const int q, const int d) const {
    const int old = quadrant(s, q);
    return s ^ side_t(old ^ R[d][old]) << 16*q;
  }

  // An exact key for two sides: per quadrant, radix 3 digits (1 for m, 2 for b) in 15 of 16 bits,
  // leaving bits 15, 31, 47, 63 spare
  uint64_t pack(const side_t m, const side_t b) const {
    uint64_t k = 0;
    for (int q = 0; q < 4; q++) k |= uint64_t(pack3[quadrant(m, q)] + 2*pack3[quadrant(b, q)]) << 16*q;
    return k;
  }

  // Does s contain a five in a row?
  bool won(const side_t s) const {
    const uint64_t x = (win_parts[0][quadrant(s, 0)] + win_parts[1][quadrant(s, 1)] +
                        win_parts[2][quadrant(s, 2)] + win_parts[3][quadrant(s, 3)]) ^ win_need;
    return ~(x | x >> 1) & 0x5555555555555555;  // Some line's count equals its need
  }

  WASM_SMALL void init() {
    memset(this, 0, sizeof(*this));
    for (const int d : {0, 1})
      for (int old = 0; old < 512; old++) {
        int quad = 0;
        for (int i = 0; i < 9; i++)
          if (old >> i & 1)
            quad |= 1 << (4 - (2*d-1)*(3*(i%3) - i/3 - 2));
        R[d][old] = quad;
      }
    // Lines and their single rotations
    side_t all[32*9], plain[32];
    int n = 0, nplain = 0;
    const int dirs[4][2] = {{1,0},{0,1},{1,1},{1,-1}};
    for (const auto& dir : dirs)
      for (int x = 0; x < 6; x++)
        for (int y = 0; y < 6; y++) {
          const int ex = x + 4*dir[0], ey = y + 4*dir[1];
          if (ex < 0 || ex >= 6 || ey < 0 || ey >= 6) continue;
          side_t line = 0;
          for (int i = 0; i < 5; i++) {
            const int cx = x + i*dir[0], cy = y + i*dir[1];
            line |= side_t(1) << (3*(cx%3) + cy%3 + 16*(2*(cx/3) + cy/3));
          }
          NON_WASM_ASSERT(nplain < 32);
          plain[nplain++] = line;
          all[n++] = line;
          for (int r = 0; r < 8; r++) all[n++] = rot(line, r >> 1, r & 1);
        }
    NON_WASM_ASSERT(nplain == 32);
    for (int q = 0; q < 512; q++) {
      int p = 0;
      for (int i = 8; i >= 0; i--) p = 3*p + (q >> i & 1);
      pack3[q] = p;
    }
    for (int i = 0; i < 32; i++)
      for (int q = 0; q < 4; q++) {
        const int seg = quadrant(plain[i], q);
        if (!seg) continue;
        win_need += uint64_t(1) << 2*i;
        for (int s = 0; s < 512; s++)
          if (!(seg & ~s)) win_parts[q][s] += uint64_t(1) << 2*i;
      }
    // Sort and deduplicate
    for (int i = 1; i < n; i++) {
      const side_t x = all[i];
      int j = i;
      for (; j > 0 && all[j-1] > x; j--) all[j] = all[j-1];
      all[j] = x;
    }
    {
      int u = 0;
      for (int i = 0; i < n; i++)
        if (!u || all[i] != all[u-1]) all[u++] = all[i];
      n = u;
    }
    NON_WASM_ASSERT(n <= nsets);
    for (int i = 0; i < 32; i++) NON_WASM_ASSERT(won(plain[i]) && !won(plain[i] & (plain[i] - 1)));
    for (int i = 0; i < n; i++) {
      sets[i] = all[i];
      real.w[i >> 6] |= uint64_t(1) << (i & 63);
      for (const side_t line : plain)
        if (all[i] == line) lines.w[i >> 6] |= uint64_t(1) << (i & 63);
    }
    // Views: view 0 is the frame, view 1+r is the frame after rotation r = (r>>1, r&1)
    lset_t cell_sets[64];
    memset(cell_sets, 0, sizeof(cell_sets));
    for (int i = 0; i < n; i++)
      for (side_t cs = sets[i]; cs; cs &= cs-1)
        cell_sets[__builtin_ctzll(cs)].w[i >> 6] |= uint64_t(1) << (i & 63);
    for (int c = 0; c < 64; c++) {
      if (!(side_t(1) << c & side_mask)) continue;  // Not a cell: rot() would index past R
      through[0][c] = cell_sets[c];
      for (int r = 0; r < 8; r++) {
        rot_cell[r][c] = rot(side_t(1) << c, r >> 1, r & 1);
        through[1+r][c] = cell_sets[__builtin_ctzll(rot_cell[r][c])];
      }
    }
    for (int i = 0; i < nsets; i++) {
      pulled[0][i] = sets[i];
      for (int r = 0; r < 8; r++) pulled[1+r][i] = rot(sets[i], r >> 1, !(r & 1));
    }
    for (int c = 0; c < 64; c++) {
      const side_t x = side_t(1) << c;
      if (!(x & side_mask)) continue;
      key_delta[0][c] = pack(x, 0);
      key_delta[1][c] = pack(0, x);
    }
  }
};

inline uint64_t nonzero(const planes_t& P, const int w) { return P.p[0][w] | P.p[1][w] | P.p[2][w]; }

inline void add_stone(const tables_t& T, side_state_t& s, const int c, const int v0, const int v1) {
  for (int v = v0; v < v1; v++) {
    const uint64_t* x = T.through[v][c].w;
    planes_t& P = s.v[v];
    for (int w = 0; w < 3; w++) {
      const uint64_t c0 = P.p[0][w] & x[w], c1 = P.p[1][w] & c0;
      P.p[0][w] ^= x[w];
      P.p[1][w] ^= c0;
      P.p[2][w] |= c1;
    }
  }
}

// Remove a counted stone (the inverse of add_stone)
inline void remove_stone(const tables_t& T, side_state_t& s, const int c, const int v0, const int v1) {
  for (int v = v0; v < v1; v++) {
    const uint64_t* x = T.through[v][c].w;
    planes_t& P = s.v[v];
    for (int w = 0; w < 3; w++) {
      const uint64_t b0 = x[w] & ~P.p[0][w], b1 = b0 & ~P.p[1][w];
      P.p[0][w] ^= x[w];
      P.p[1][w] ^= b0;
      P.p[2][w] ^= b1;
    }
  }
}

// The parent's counts plus one stone, written directly
inline void child_state(const tables_t& T, side_state_t& out, const side_state_t& in, const int c) {
  for (int v = 0; v < 9; v++) {
    const uint64_t* x = T.through[v][c].w;
    const planes_t& P = in.v[v];
    planes_t& Q = out.v[v];
    for (int w = 0; w < 3; w++) {
      const uint64_t c0 = P.p[0][w] & x[w], c1 = P.p[1][w] & c0;
      Q.p[0][w] = P.p[0][w] ^ x[w];
      Q.p[1][w] = P.p[1][w] ^ c0;
      Q.p[2][w] = P.p[2][w] | c1;
    }
  }
}

// The Maker to move wins at once, given that view 0 says it might: by a placement five, or through
// a rotation that doesn't also complete a Breaker line
bool refined_win(const tables_t& T, const side_state_t& M, const side_state_t& B) {
  for (int v = 0; v < 9; v++) {
    const planes_t &P = M.v[v], &Q = B.v[v];
    uint64_t win = 0, bfive = 0;
    for (int w = 0; w < 3; w++) {
      win |= P.p[2][w] & ~nonzero(Q, w) & T.lines.w[w];
      bfive |= Q.p[0][w] & Q.p[2][w] & T.lines.w[w];
    }
    // A placement five wins outright.  (A Breaker five in view 0 means the Breaker has already won,
    // so the answer there doesn't matter.)
    if (win && (!v || !bfive)) return true;
  }
  return false;
}

// Outer memo entry: the fewest free replies known to hold and the most known to fail
struct hold_entry_t {
  uint64_t key;  // pack(x, y) | valid bit 31 | role bit 15 (set for the player to move)
  int8_t hold, fail;
};

void* tie_alloc(const size_t size) {
#ifdef __wasm__
  return malloc(size);
#else
  void* p = std::malloc(size);
  NON_WASM_ASSERT(p);
  return p;
#endif
}

void tie_free(void* p) {
#ifdef __wasm__
  free(p);
#else
  std::free(p);
#endif
}

#ifndef __wasm__
double wall_clock() { return wall_time().seconds(); }
#endif

}  // namespace

// Plain data, set up by init, so that it can live in malloced memory
struct tie_prover_t::impl_t {
  tables_t T;

  // Static memo: an entry is pack(m, b) (60 bits), with a valid bit in bit 31 and the result (Maker
  // wins) in bit 15, the spare top bits of the first two quadrant fields.  Buckets of 4 entries.
  int memo_bits;
  uint64_t* memo;
  static constexpr uint64_t result_bit = uint64_t(1) << 15;
  static constexpr uint64_t valid_bit = uint64_t(1) << 31;

  // Outer memo, also in buckets of 4
  int hold_bits;
  hold_entry_t* holds;

  // History heuristic for the outer search, indexed by 8*cell + rotation
  int64_t mover_history[512], other_history[512];

  tie_clock_t clock;
  double deadline;
  bool timeout;
  int64_t nodes;
  int64_t time_check_mask;
#ifdef __wasm__
  static constexpr bool paranoid = false;  // Paranoid checks are for native tests, so wasm drops them
#else
  bool paranoid;
#endif

  WASM_SMALL void init(const int bits) {
    NON_WASM_ASSERT(5 <= bits && bits <= 30);
    T.init();
    memo_bits = bits;
    hold_bits = bits - 3;
    memo = static_cast<uint64_t*>(tie_alloc(sizeof(uint64_t) << memo_bits));
    holds = static_cast<hold_entry_t*>(tie_alloc(sizeof(hold_entry_t) << hold_bits));
    clear();
#ifdef __wasm__
    clock = nullptr;
#else
    clock = wall_clock;
#endif
    deadline = 0;
    timeout = false;
    nodes = 0;
    time_check_mask = 1023;
#ifndef __wasm__
    paranoid = false;
#endif
  }

  void release() {
    tie_free(memo);
    tie_free(holds);
  }

  void clear() {
    memset(memo, 0, sizeof(uint64_t) << memo_bits);
    memset(holds, 0, sizeof(hold_entry_t) << hold_bits);
    clear_history();
  }

  void clear_history() {
    memset(mover_history, 0, sizeof(mover_history));
    memset(other_history, 0, sizeof(other_history));
  }

  void start_call(const double seconds) {
    timeout = false;
    deadline = clock ? clock() + seconds : 0;
  }

  static uint64_t hash(const uint64_t key) { return key * 0x9e3779b97f4a7c15ull; }

  uint64_t* bucket(const uint64_t key) {
    return memo + ((hash(key) >> (64 - memo_bits)) & ~uint64_t(3));
  }

  bool memo_get(const uint64_t key, bool& result) {
    const uint64_t* e = bucket(key);
    const uint64_t want = key | valid_bit;
    for (int i = 0; i < 4; i++)
      if ((e[i] & ~result_bit) == want) {
        result = e[i] & result_bit;
        return true;
      }
    return false;
  }

  void memo_put(const uint64_t key, const bool result) {
    uint64_t* e = bucket(key);
    int i = 0;
    for (; i < 4; i++)
      if (!e[i] || (e[i] & ~result_bit) == (key | valid_bit)) break;  // Empty or same
    if (i == 4) i = (key >> 17 ^ key >> 41) & 3;  // Full: replace a pseudorandom entry
    e[i] = key | valid_bit | (result ? result_bit : 0);
  }

  uint64_t hold_key(const side_t x, const side_t y, const bool mover) const {
    return T.pack(x, y) | valid_bit | (mover ? result_bit : 0);
  }

  hold_entry_t* hold_bucket(const uint64_t key) {
    return holds + ((hash(key) >> (64 - hold_bits)) & ~uint64_t(3));
  }

  // Known answer for depth d, if any (1 holds, 0 fails, -1 unknown), and the most free replies known
  // to fail
  int hold_get(const uint64_t key, const int d, int& fail) {
    const hold_entry_t* e = hold_bucket(key);
    fail = -1;
    for (int i = 0; i < 4; i++)
      if (e[i].key == key) {
        fail = e[i].fail;
        return e[i].hold <= d ? 1 : d <= e[i].fail ? 0 : -1;
      }
    return -1;
  }

  void hold_put(const uint64_t key, const int d, const bool result) {
    hold_entry_t* e = hold_bucket(key);
    int i = 0;
    for (; i < 4; i++)
      if (!e[i].key || e[i].key == key) break;
    if (i == 4) i = (key >> 17 ^ key >> 41) & 3;
    if (e[i].key != key) e[i] = hold_entry_t{key, 127, -1};
    if (result && d < e[i].hold) e[i].hold = d;
    if (!result && d > e[i].fail) e[i].fail = d;
  }

  void check_time() {
    if (!(++nodes & time_check_mask) && clock && clock() > deadline) timeout = true;
  }

  // Add a stone to the frame view counts
  void add_view0(planes_t& P, const int c) const {
    const uint64_t* x = T.through[0][c].w;
    for (int w = 0; w < 3; w++) {
      const uint64_t c0 = P.p[0][w] & x[w], c1 = P.p[1][w] & c0;
      P.p[0][w] ^= x[w];
      P.p[1][w] ^= c0;
      P.p[2][w] |= c1;
    }
  }

  // Counts for a side from scratch
  side_state_t scratch_state(const side_t s) const {
    side_state_t x;
    memset(&x, 0, sizeof(x));
    for (side_t cs = s; cs; cs &= cs-1) add_stone(T, x, __builtin_ctzll(cs), 0, 9);
    return x;
  }

  void verify(const side_state_t& M, const side_state_t& B, const side_t m, const side_t b, const uint64_t key) const {
    const auto sm = scratch_state(m), sb = scratch_state(b);
    bool same = key == T.pack(m, b);
    for (int v = 0; v < 9; v++)
      for (int i = 0; i < 3; i++)
        for (int w = 0; w < 3; w++)
          same &= sm.v[v].p[i][w] == M.v[v].p[i][w] && sb.v[v].p[i][w] == B.v[v].p[i][w];
#ifndef __wasm__
    // Our win table agrees with pentago/base
    for (int r = 0; r < 9; r++) {
      const side_t rm = r < 8 ? T.rot(m, r >> 1, r & 1) : m, rb = r < 8 ? T.rot(b, r >> 1, r & 1) : b;
      same &= T.won(rm) == pentago::won(rm) && T.won(rb) == pentago::won(rb);
    }
#endif
    (void)same;  // Unused in wasm, where paranoid mode is compiled out
    NON_WASM_ASSERT(same);
  }

  // Does the Maker win at once, by direct search over placements and rotations?
  bool slow_win(const side_t m, const side_t b) const {
    for (side_t ps = side_mask & ~(m|b); ps; ps &= ps-1) {
      const side_t mp = m | (ps & -ps);
      if (T.won(mp)) return true;
      for (int r = 0; r < 8; r++)
        if (T.won(T.rot(mp, r >> 1, r & 1)) && !T.won(T.rot(b, r >> 1, r & 1))) return true;
    }
    return false;
  }

  // Order empty cells by Erdős–Selfridge weight of the Maker's live sets
  int order_cells(const side_t empty, const uint64_t* two, const uint64_t* three, const uint64_t* four,
                  int* cells) const {
    int weight[64], nc = 0;
    for (side_t cs = empty; cs; cs &= cs-1) {
      const int c = __builtin_ctzll(cs);
      const uint64_t* x = T.through[0][c].w;
      int wt = 0;
      for (int w = 0; w < 3; w++)
        wt += 4*popcount(two[w] & x[w]) + 2*popcount(three[w] & x[w]) + popcount(four[w] & x[w]);
      weight[c] = wt;
      cells[nc++] = c;
    }
    for (int i = 1; i < nc; i++) {
      const int c = cells[i], wc = weight[c];
      int j = i;
      for (; j > 0 && weight[cells[j-1]] < wc; j--) cells[j] = cells[j-1];
      cells[j] = c;
    }
    return nc;
  }

  // What the frame views alone say, with the Maker to move and k placements left: 1 if the Maker
  // wins at once, 0 if the Breaker holds, 2 if the Maker might win at once but some rotations
  // complete fives for both sides (see refined_win), and -1 if we must search.
  //
  // The Breaker holds if k == 1 and the Maker can't win now, or by Erdős–Selfridge: if the live sets
  // the Maker can still fill have sum 2^-missing < 1/2, the Breaker wins by always taking the cell
  // of greatest weight.  That cell blocks the Maker's threat if it has one (a threat weighs 1/2, and
  // after the Maker's move the sum is below 1), so it is a legal reply here, and the restrictions on
  // Maker rotations only help the Breaker.
  int quick(const planes_t& M0, const planes_t& B0, const int k) const {
    uint64_t live[3], win = 0, covers = 0;
    for (int w = 0; w < 3; w++) {
      live[w] = ~nonzero(B0, w) & T.real.w[w];
      win |= M0.p[2][w] & live[w];
      covers |= B0.p[0][w] & B0.p[2][w];
    }
    if (win) return covers ? 2 : 1;
    if (k <= 1) return 0;
    // Sets missing 2, 3, 4, 5 cells, in order of weight, stopping once the sum reaches 1/2
    int potential = 0;
    for (int missing = 2; missing <= 5 && missing <= k; missing++) {
      const int c = 5 - missing;  // Maker count
      for (int w = 0; w < 3; w++) {
        const uint64_t p0 = M0.p[0][w], p1 = M0.p[1][w];
        potential += popcount((c & 1 ? p0 : ~p0) & (c & 2 ? p1 : ~p1) & ~M0.p[2][w] & live[w]);
      }
      if (potential << (5 - missing) >= 16) return -1;  // potential is in units of 2^-missing
      potential <<= 1;
    }
    return 0;
  }

  // Can the Maker (stones m, to move, k placements left) beat the undo strategy of the Breaker?
  // key = pack(m, b).  Returns false on timeout, so callers must check.
  bool maker(const side_state_t& M, const side_state_t& B, const side_t m, const side_t b, const int k,
             const uint64_t key) {
    if (k <= 0) return false;
    if (paranoid) verify(M, B, m, b, key);
    {
      bool r;
      if (memo_get(key, r)) return r;
    }
    int q = quick(M.v[0], B.v[0], k);
    if (q == 2) q = refined_win(T, M, B) ? 1 : k <= 1 ? 0 : -1;
    if (paranoid) NON_WASM_ASSERT((q == 1) == slow_win(m, b));
    if (q >= 0) {
      memo_put(key, q);
      return q;
    }
    const planes_t& M0 = M.v[0];
    uint64_t two[3], three[3], four[3];
    for (int w = 0; w < 3; w++) {
      const uint64_t live = ~nonzero(B.v[0], w);
      two[w] = M0.p[0][w] & M0.p[1][w] & ~M0.p[2][w] & live;
      three[w] = ~M0.p[0][w] & M0.p[1][w] & ~M0.p[2][w] & live;
      four[w] = M0.p[0][w] & ~M0.p[1][w] & ~M0.p[2][w] & live;
    }
    check_time();
    if (timeout) return false;

    // Missing pairs of the Maker's threat sets
    side_t pairs[nsets];
    int npairs = 0;
    for (int w = 0; w < 3; w++)
      for (uint64_t t = two[w]; t; t &= t-1)
        pairs[npairs++] = T.sets[64*w + __builtin_ctzll(t)] & ~m;

    // Breaker's immediate chances after each Maker rotation r.  If some rotation leaves none, every
    // placement is safe; otherwise a placement is safe only if it hits every chance of some rotation.
    side_t inter[8];
    bool constrained = true;
    {
      uint64_t near[8][3];
      for (int r = 0; r < 8 && constrained; r++) {
        uint64_t any = 0;
        for (int w = 0; w < 3; w++) any |= near[r][w] = B.v[1+r].p[2][w] & ~nonzero(M.v[1+r], w);
        constrained = any;
      }
      if (constrained)
        for (int r = 0; r < 8; r++) {
          inter[r] = ~side_t(0);
          for (int w = 0; w < 3; w++)
            for (uint64_t t = near[r][w]; t; t &= t-1)
              inter[r] &= T.pulled[1+r][64*w + __builtin_ctzll(t)];
        }
    }

    const side_t empty = side_mask & ~(m|b);
    int cells[36];
    const int nc = order_cells(empty, two, three, four, cells);
    bool result = false;
    for (int pi = 0; pi < nc && !result && !timeout; pi++) {
      const side_t p = side_t(1) << cells[pi];
      if (constrained) {
        bool safe = false;
        for (int r = 0; r < 8 && !safe; r++) safe = p & inter[r];
        if (!safe) continue;
      }
      side_t threats = 0;
      for (int i = 0; i < npairs; i++)
        if (pairs[i] & p) {
          const side_t t = pairs[i] & ~p;
          if (threats & ~t) {
            threats |= t;
            break;
          }
          threats |= t;
        }
      if (threats & (threats - 1)) {  // Two threats: the Breaker can block only one
        result = true;
        break;
      }
      // With two placements left, the last one wins only if this one made two threats
      if (k == 2) continue;
      side_state_t M2;
      child_state(T, M2, M, cells[pi]);
      const uint64_t key2 = key + T.key_delta[0][cells[pi]];
      const side_t replies = threats ? threats : empty & ~p;
      // Settle what replies we can from the frame views alone, stopping at any after which the
      // Breaker holds.  The rest need the memo or a search.
      int open[36], nopen = 0;
      bool all = true;
      for (int ci = 0; ci < nc && all; ci++) {
        const int c = cells[ci];
        if (!(replies >> c & 1)) continue;
        planes_t B20 = B.v[0];
        add_view0(B20, c);
        const int cq = quick(M2.v[0], B20, k - 1);
        if (paranoid && (cq == 0 || cq == 1)) NON_WASM_ASSERT((cq == 1) == slow_win(m | p, b | side_t(1) << c));
        if (cq == 0) all = false;
        else if (cq != 1) open[nopen++] = c;
      }
      if (all) {
        for (int i = 0; i < nopen; i++) __builtin_prefetch(bucket(key2 + T.key_delta[1][open[i]]));
        for (int i = 0; i < nopen && all && !timeout; i++) {
          const uint64_t ckey = key2 + T.key_delta[1][open[i]];
          bool r;
          if (memo_get(ckey, r)) {
            all = r;
            continue;
          }
          side_state_t B2;
          child_state(T, B2, B, open[i]);
          all = maker(M2, B2, m | p, b | side_t(1) << open[i], k - 1, ckey);
        }
      }
      result = all && !timeout;
    }
    if (!timeout) memo_put(key, result);
    return result;
  }

  // Does the Breaker (stones b, not to move) hold by pure undo against m?
  bool breaker_holds(const side_t m, const side_t b) {
    const uint64_t key = T.pack(m, b);
    {
      bool r;
      if (memo_get(key, r)) return !r;
    }
    const int k = (36 - popcount(m|b) + 1) / 2;  // Maker's remaining placements
    // Build the frame view first: most solves end at the root's immediate checks
    side_state_t M, B;
    memset(&M.v[0], 0, sizeof(planes_t));
    memset(&B.v[0], 0, sizeof(planes_t));
    for (side_t cs = m; cs; cs &= cs-1) add_stone(T, M, __builtin_ctzll(cs), 0, 1);
    for (side_t cs = b; cs; cs &= cs-1) add_stone(T, B, __builtin_ctzll(cs), 0, 1);
    const int q = quick(M.v[0], B.v[0], k);
    bool lose;
    if (q == 0 || q == 1) {
      lose = q;
      memo_put(key, lose);
    } else {
      memset(&M.v[1], 0, 8*sizeof(planes_t));
      memset(&B.v[1], 0, 8*sizeof(planes_t));
      for (side_t cs = m; cs; cs &= cs-1) add_stone(T, M, __builtin_ctzll(cs), 1, 9);
      for (side_t cs = b; cs; cs &= cs-1) add_stone(T, B, __builtin_ctzll(cs), 1, 9);
      lose = maker(M, B, m, b, k, key);
    }
    return !lose && !timeout;  // A timed out search proves nothing
  }

  // Order moves by decreasing history score, stably.  Insertion sort is fast here since most scores
  // are zero.
  static void history_sort(const int64_t* h, side_t* X, side_t* Y, int* id, const int n) {
    for (int i = 1; i < n; i++) {
      const side_t x = X[i], y = Y[i];
      const int k = id[i];
      const int64_t hk = h[k];
      int j = i;
      for (; j > 0 && h[id[j-1]] < hk; j--) X[j] = X[j-1], Y[j] = Y[j-1], id[j] = id[j-1];
      X[j] = x, Y[j] = y, id[j] = k;
    }
  }

  // Player to move (stones a) holds: some first move, after which it holds as the player not to
  // move with up to d free replies
  bool mover_holds(const side_t a, const side_t o, const int d) {
    check_time();
    if (timeout) return false;
    const uint64_t key = hold_key(a, o, true);
    int fail;
    if (const int known = hold_get(key, d, fail); known >= 0) return known;
    side_t ar[8], or_[8];
    bool lost[8];  // The rotation alone completes a line for o
    for (int r = 0; r < 8; r++) {
      ar[r] = T.rot(a, r >> 1, r & 1), or_[r] = T.rot(o, r >> 1, r & 1);
      lost[r] = T.won(or_[r]);
    }
    side_t A[288], O[288];
    int id[288], n = 0;
    for (side_t ms = side_mask & ~(a|o); ms; ms &= ms-1) {
      const int c = __builtin_ctzll(ms);
      if (T.won(a | side_t(1) << c)) return true;
      for (int r = 0; r < 8; r++) {
        const side_t a2 = ar[r] | T.rot_cell[r][c];
        if (T.won(a2)) return true;  // Win or tie
        if (lost[r]) continue;
        if ((a2|or_[r]) == side_mask) return true;  // Full board: tie
        A[n] = a2, O[n] = or_[r], id[n] = 8*c + r, n++;
      }
    }

    // Line counts of both sides after each rotation, built lazily: the frame view to settle first
    // moves the frame decides (see quick), all views to search the rest without rebuilding them
    const int d0 = fail + 1;  // Every first move is known to fail with up to `fail` free replies
    const int k = (36 - popcount(a|o)) / 2;  // The opponent's placements after our move
    side_state_t Ar[8], Or[8], Ab, Ob;  // Ab, Ob unrotated
    int built[8] = {0}, base = 0;  // Views built
    const auto build = [&](const int r, const int views) {
      if (built[r] >= views) return;
      if (base < views) {
        memset(&Ab.v[base], 0, (views - base)*sizeof(planes_t));
        memset(&Ob.v[base], 0, (views - base)*sizeof(planes_t));
        for (side_t cs = a; cs; cs &= cs-1) add_stone(T, Ab, __builtin_ctzll(cs), base, views);
        for (side_t cs = o; cs; cs &= cs-1) add_stone(T, Ob, __builtin_ctzll(cs), base, views);
        base = views;
      }
      // A rotation moves the stones of one quadrant
      const int v0 = built[r];
      memcpy(&Ar[r].v[v0], &Ab.v[v0], (views - v0)*sizeof(planes_t));
      memcpy(&Or[r].v[v0], &Ob.v[v0], (views - v0)*sizeof(planes_t));
      for (side_t cs = a & ~ar[r]; cs; cs &= cs-1) remove_stone(T, Ar[r], __builtin_ctzll(cs), v0, views);
      for (side_t cs = ar[r] & ~a; cs; cs &= cs-1) add_stone(T, Ar[r], __builtin_ctzll(cs), v0, views);
      for (side_t cs = o & ~or_[r]; cs; cs &= cs-1) remove_stone(T, Or[r], __builtin_ctzll(cs), v0, views);
      for (side_t cs = or_[r] & ~o; cs; cs &= cs-1) add_stone(T, Or[r], __builtin_ctzll(cs), v0, views);
      built[r] = views;
    };
    int kept = 0, result = -1;
    for (int i = 0; i < n; i++) {
      const int r = id[i] & 7;
      build(r, 1);
      planes_t A0 = Ar[r].v[0];
      add_view0(A0, __builtin_ctzll(A[i] & ~ar[r]));
      const int q = quick(Or[r].v[0], A0, k);
      if (q == 1) continue;  // The opponent wins at once, whatever our free replies
      if (q == 0) {  // We hold by undo
        result = d0;
        mover_history[id[i]] += int64_t(1) << 2*d0;
        break;
      }
      A[kept] = A[i], O[kept] = O[i], id[kept] = id[i], kept++;
    }
    n = kept;

    if (result < 0) {
      history_sort(mover_history, A, O, id, n);
      for (int dd = d0; dd <= d && !timeout && result < 0; dd++)
        for (int i = 0; i < n && !timeout && result < 0; i++) {
          bool holds;
          if (dd) {
            holds = other_holds(O[i], A[i], dd);
          } else {
            const int r = id[i] & 7;
            build(r, 9);
            side_state_t B;
            child_state(T, B, Ar[r], __builtin_ctzll(A[i] & ~ar[r]));
            holds = !maker(Or[r], B, O[i], A[i], k, T.pack(O[i], A[i])) && !timeout;
          }
          if (holds) {
            result = dd;
            mover_history[id[i]] += int64_t(1) << 2*dd;
          }
        }
    }
    if (timeout) return false;
    hold_put(key, result >= 0 ? result : d, result >= 0);
    return result >= 0;
  }

  // Player not to move (stones b) holds against m: by undo now, or (if d > 0) by answering every
  // move of m with a free move of its own and holding with d-1
  bool other_holds(const side_t m, const side_t b, const int d) {
    if (breaker_holds(m, b)) return true;
    if (timeout || d <= 0) return false;
    check_time();
    if (timeout) return false;
    const uint64_t key = hold_key(m, b, false);
    int fail;
    if (const int known = hold_get(key, d, fail); known >= 0) return known;
    bool result = true;
    side_t Bs[288], Ms[288];
    int id[288], n = 0;
    for (side_t ms = side_mask & ~(m|b); ms && result; ms &= ms-1) {
      const side_t mp = m | (ms & -ms);
      if (T.won(mp)) {
        result = false;
        break;
      }
      for (int r = 0; r < 8 && result; r++) {
        const side_t m2 = T.rot(mp, r >> 1, r & 1), b2 = T.rot(b, r >> 1, r & 1);
        if (T.won(b2)) continue;  // Win or tie for b
        if (T.won(m2)) {
          result = false;
          break;
        }
        if ((m2|b2) == side_mask) continue;  // Full board: tie
        Bs[n] = b2, Ms[n] = m2, id[n] = 8*__builtin_ctzll(ms) + r, n++;
      }
    }
    if (result) {
      history_sort(other_history, Ms, Bs, id, n);
      for (int i = 0; i < n && result && !timeout; i++)
        if (!(result = mover_holds(Bs[i], Ms[i], d - 1)))
          other_history[id[i]] += int64_t(1) << 2*d;
    }
    if (timeout) return false;
    hold_put(key, d, result);
    return result;
  }
};

WASM_SMALL tie_prover_t::tie_prover_t(const int memo_bits)
  : impl(static_cast<impl_t*>(tie_alloc(sizeof(impl_t)))) {
  impl->init(memo_bits);
}

tie_prover_t::~tie_prover_t() {
  impl->release();
  tie_free(impl);
}

void tie_prover_t::clear() {
  impl->clear();
}

void tie_prover_t::set_clock(const tie_clock_t clock) {
  impl->clock = clock;
}

WASM_SMALL tie_result_t tie_prover_t::prove(const high_board_t board, const int max_depth, const double seconds) {
  NON_WASM_ASSERT(!board.middle() && !board.done() && 0 <= max_depth && max_depth < 127);
  impl_t& I = *impl;
  I.clear_history();
  I.start_call(seconds);
  const side_t a = board.side(board.turn()), o = board.side(!board.turn());
  tie_result_t result;
  for (int d = 0; d <= max_depth && !I.timeout && (result.mover < 0 || result.other < 0); d++) {
    if (result.mover < 0 && I.mover_holds(a, o, d)) result.mover = d;
    if (!I.timeout && result.other < 0 && I.other_holds(a, o, d)) result.other = d;
  }
  result.timeout = I.timeout;
  return result;
}

bool tie_prover_t::mover_holds(const side_t to_move, const side_t other, const int depth, const double seconds) {
  NON_WASM_ASSERT(0 <= depth && depth < 127);
  impl_t& I = *impl;
  I.start_call(seconds);
  return I.mover_holds(to_move, other, depth) && !I.timeout;
}

bool tie_prover_t::other_holds(const side_t to_move, const side_t other, const int depth, const double seconds) {
  NON_WASM_ASSERT(0 <= depth && depth < 127);
  impl_t& I = *impl;
  I.start_call(seconds);
  return I.other_holds(to_move, other, depth) && !I.timeout;
}

void tie_prover_t::set_time_check_bits(const int bits) {
  NON_WASM_ASSERT(0 <= bits && bits < 63);
  impl->time_check_mask = (int64_t(1) << bits) - 1;
}

void tie_prover_t::set_paranoid(const bool paranoid) {
#ifdef __wasm__
  NON_WASM_ASSERT(!paranoid);
#else
  impl->paranoid = paranoid;
#endif
}

bool tie_prover_t::undo_holds(const high_board_t board) {
  NON_WASM_ASSERT(!board.middle() && !board.done());
  impl_t& I = *impl;
  I.start_call(1e9);
  return I.breaker_holds(board.side(board.turn()), board.side(!board.turn()));
}

}  // namespace pentago
