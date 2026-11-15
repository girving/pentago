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
// Placements and replies are ordered by Erdős–Selfridge weight (live Maker sets through the cell,
// weighted 2^-missing).  Positions are memoized exactly in 8-byte entries.

#include "pentago/cascade/tie.h"
#include "pentago/base/score.h"
#include "pentago/utility/array.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/popcount.h"
#include "pentago/utility/wall_time.h"
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>
namespace pentago {

using std::unordered_map;
using std::vector;

namespace {

const int nsets = 160;  // Rotated lines, padded

struct lset_t { uint64_t w[3]; };          // A subset of the 160 sets
struct planes_t { uint64_t p[3][3]; };     // Bit-sliced counts: [plane][word]
struct side_state_t { planes_t v[9]; };    // Counts for each view

struct tables_t {
  uint16_t R[2][512] = {};         // Quadrant rotations in both directions, matching high_board_t::rotate
  side_t sets[nsets] = {};         // Cells of each rotated line, in frame coordinates
  side_t pulled[9][nsets] = {};    // Set i of view v pulled back to the frame
  lset_t through[9][64] = {};      // Sets of view v containing each frame cell, after it moves with the view
  uint64_t key_delta[2][64] = {};  // pack(m, b) increments for a Maker or Breaker stone

  side_t rot(const side_t s, const int q, const int d) const {
    const int old = quadrant(s, q);
    return s ^ side_t(old ^ R[d][old]) << 16*q;
  }

  tables_t() {
    for (const int d : {0, 1})
      for (int old = 0; old < 512; old++) {
        int quad = 0;
        for (int i = 0; i < 9; i++)
          if (old >> i & 1)
            quad |= 1 << (4 - (2*d-1)*(3*(i%3) - i/3 - 2));
        R[d][old] = quad;
      }
    // Lines and their single rotations
    vector<side_t> all;
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
          GEODE_ASSERT(won(line));
          all.push_back(line);
          for (int r = 0; r < 8; r++) all.push_back(rot(line, r >> 1, r & 1));
        }
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    GEODE_ASSERT(all.size() <= size_t(nsets));
    for (int i = 0; i < int(all.size()); i++) sets[i] = all[i];
    // Views: view 0 is the frame, view 1+r is the frame after rotation r = (r>>1, r&1)
    lset_t cell_sets[64] = {};
    for (int i = 0; i < int(all.size()); i++)
      for (side_t cs = sets[i]; cs; cs &= cs-1)
        cell_sets[__builtin_ctzll(cs)].w[i >> 6] |= uint64_t(1) << (i & 63);
    for (int c = 0; c < 64; c++) {
      if (!(side_t(1) << c & side_mask)) continue;  // Not a cell: rot() would index past R
      through[0][c] = cell_sets[c];
      for (int r = 0; r < 8; r++)
        through[1+r][c] = cell_sets[__builtin_ctzll(rot(side_t(1) << c, r >> 1, r & 1))];
    }
    for (int i = 0; i < nsets; i++) {
      pulled[0][i] = sets[i];
      for (int r = 0; r < 8; r++) pulled[1+r][i] = rot(sets[i], r >> 1, !(r & 1));
    }
    for (int c = 0; c < 64; c++) {
      const side_t x = side_t(1) << c;
      if (!(x & side_mask)) continue;
      key_delta[0][c] = pack(x, side_t(0));
      key_delta[1][c] = pack(side_t(0), x);
    }
  }
};

const tables_t& tables() {
  static const tables_t t;
  return t;
}

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

// Outer memo key: d >= 0 for other_holds(x, y, d), -1-d for mover_holds(x, y, d)
struct hold_key_t {
  side_t x, y;
  int d;
  bool operator==(const hold_key_t& k) const { return x == k.x && y == k.y && d == k.d; }
};
struct hold_hash_t {
  size_t operator()(const hold_key_t& k) const {
    return k.x * 0x9e3779b97f4a7c15ull ^ k.y * 0xc2b2ae3d27d4eb4full ^ size_t(k.d) * 0x165667b19e3779f9ull;
  }
};

}  // namespace

struct tie_prover_t::impl_t {
  const tables_t& T;

  // Static memo: an entry is pack(m, b) (60 bits), with the result (Maker wins) in bit 15 and a 3-bit
  // board stamp in bits 31, 47, 63, the spare top bit of each quadrant field.  Buckets of 4 entries.
  const int memo_bits;
  Array<uint64_t> memo;
  uint64_t stamp = 0;
  static constexpr uint64_t result_bit = uint64_t(1) << 15;
  static constexpr uint64_t stamp_mask = uint64_t(1) << 31 | uint64_t(1) << 47 | uint64_t(1) << 63;

  unordered_map<hold_key_t,bool,hold_hash_t> holds;

  // History heuristic for the outer search, indexed by 8*cell + rotation
  int64_t mover_history[512], other_history[512];

  wall_time_t deadline;
  bool timeout = false;
  int64_t nodes = 0;
  int64_t time_check_mask = 1023;
  bool paranoid = false;

  explicit impl_t(const int memo_bits)
    : T(tables()), memo_bits(memo_bits), memo(size_t(1) << memo_bits) {
    GEODE_ASSERT(2 <= memo_bits && memo_bits <= 30);
  }

  void new_board(const double seconds) {
    if (++stamp == 8) {
      memo.fill(0);
      stamp = 1;
    }
    holds.clear();
    memset(mover_history, 0, sizeof(mover_history));
    memset(other_history, 0, sizeof(other_history));
    timeout = false;
    nodes = 0;
    deadline = wall_time();
    deadline += wall_time_t(int64_t(1e6 * std::min(seconds, 1e9)));
  }

  static uint64_t stamp_bits(const uint64_t s) {
    return (s & 1) << 31 | (s >> 1 & 1) << 47 | (s >> 2 & 1) << 63;
  }

  uint64_t* bucket(const uint64_t key) {
    return memo.data() + ((key * 0x9e3779b97f4a7c15ull >> (64 - memo_bits)) & ~uint64_t(3));
  }

  bool memo_get(const uint64_t key, bool& result) {
    const uint64_t* e = bucket(key);
    const uint64_t want = key | stamp_bits(stamp);
    for (int i = 0; i < 4; i++)
      if ((e[i] & ~result_bit) == want) {
        result = e[i] & result_bit;
        return true;
      }
    return false;
  }

  void memo_put(const uint64_t key, const bool result) {
    uint64_t* e = bucket(key);
    const uint64_t sb = stamp_bits(stamp);
    int i = 0;
    for (; i < 4; i++)
      if ((e[i] & stamp_mask) != sb || (e[i] & ~result_bit) == (key | sb)) break;  // Stale, empty or same
    if (i == 4) i = (key >> 17 ^ key >> 41) & 3;  // Full: replace a pseudorandom entry
    e[i] = key | sb | (result ? result_bit : 0);
  }

  void check_time() {
    if (!(++nodes & time_check_mask) && wall_time() > deadline) timeout = true;
  }

  // Counts for a side from scratch
  side_state_t scratch_state(const side_t s) const {
    side_state_t x;
    memset(&x, 0, sizeof(x));
    for (side_t cs = s; cs; cs &= cs-1) add_stone(T, x, __builtin_ctzll(cs), 0, 9);
    return x;
  }

  void verify(const side_state_t& M, const side_state_t& B, const side_t m, const side_t b, const uint64_t key) const {
    GEODE_ASSERT(key == pack(m, b));
    const auto sm = scratch_state(m), sb = scratch_state(b);
    GEODE_ASSERT(!memcmp(&sm, &M, sizeof(sm)) && !memcmp(&sb, &B, sizeof(sb)));
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
    const planes_t& M0 = M.v[0];
    uint64_t win = 0, two[3], three[3], four[3];
    for (int w = 0; w < 3; w++) {
      const uint64_t live = ~nonzero(B.v[0], w);
      win |= M0.p[2][w] & live;
      two[w] = M0.p[0][w] & M0.p[1][w] & ~M0.p[2][w] & live;
      three[w] = ~M0.p[0][w] & M0.p[1][w] & ~M0.p[2][w] & live;
      four[w] = M0.p[0][w] & ~M0.p[1][w] & ~M0.p[2][w] & live;
    }
    if (win || k == 1) {
      memo_put(key, bool(win));
      return win;
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
      for (int ci = 0; ci < nc; ci++)
        if (replies >> cells[ci] & 1)
          __builtin_prefetch(bucket(key2 + T.key_delta[1][cells[ci]]));
      bool all = true;
      for (int ci = 0; ci < nc && all && !timeout; ci++) {
        if (!(replies >> cells[ci] & 1)) continue;
        const uint64_t ckey = key2 + T.key_delta[1][cells[ci]];
        bool r;
        if (memo_get(ckey, r)) {
          all = r;
          continue;
        }
        side_state_t B2;
        child_state(T, B2, B, cells[ci]);
        all = maker(M2, B2, m | p, b | side_t(1) << cells[ci], k - 1, ckey);
      }
      result = all && !timeout;
    }
    if (!timeout) memo_put(key, result);
    return result;
  }

  // Does the Breaker (stones b, not to move) hold by pure undo against m?
  bool breaker_holds(const side_t m, const side_t b) {
    const uint64_t key = pack(m, b);
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
    uint64_t win = 0;
    for (int w = 0; w < 3; w++) win |= M.v[0].p[2][w] & ~nonzero(B.v[0], w);
    bool lose;
    if (win || k <= 1) {
      lose = win;
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

  static void history_sort(const int64_t* h, side_t* X, side_t* Y, int* id, const int n) {
    int idx[144];
    for (int i = 0; i < n; i++) idx[i] = i;
    std::stable_sort(idx, idx + n, [&](const int i, const int j) { return h[id[i]] > h[id[j]]; });
    side_t X2[144], Y2[144];
    int id2[144];
    for (int i = 0; i < n; i++) X2[i] = X[idx[i]], Y2[i] = Y[idx[i]], id2[i] = id[idx[i]];
    for (int i = 0; i < n; i++) X[i] = X2[i], Y[i] = Y2[i], id[i] = id2[i];
  }

  // Player to move (stones a) holds: some first move, after which it holds as the player not to
  // move with up to d free replies
  bool mover_holds(const side_t a, const side_t o, const int d) {
    side_t A[144], O[144];
    int id[144], n = 0;
    for (side_t ms = side_mask & ~(a|o); ms; ms &= ms-1) {
      const side_t ap = a | (ms & -ms);
      if (won(ap)) return true;
      for (int r = 0; r < 8; r++) {
        const side_t a2 = T.rot(ap, r >> 1, r & 1), o2 = T.rot(o, r >> 1, r & 1);
        if (won(a2)) return true;  // Win or tie
        if (won(o2)) continue;
        if ((a2|o2) == side_mask) return true;  // Full board: tie
        A[n] = a2, O[n] = o2, id[n] = 8*__builtin_ctzll(ms) + r, n++;
      }
    }
    const hold_key_t key{a, o, -1-d};
    if (const auto it = holds.find(key); it != holds.end()) return it->second;
    history_sort(mover_history, A, O, id, n);
    bool result = false;
    for (int dd = 0; dd <= d && !timeout && !result; dd++)
      for (int i = 0; i < n && !timeout && !result; i++)
        if (other_holds(O[i], A[i], dd)) {
          result = true;
          mover_history[id[i]] += int64_t(1) << 2*dd;
        }
    if (!timeout) holds[key] = result;
    return result && !timeout;
  }

  // Player not to move (stones b) holds against m: by undo now, or (if d > 0) by answering every
  // move of m with a free move of its own and holding with d-1
  bool other_holds(const side_t m, const side_t b, const int d) {
    if (breaker_holds(m, b)) return true;
    if (timeout || d <= 0) return false;
    const hold_key_t key{m, b, d};
    if (const auto it = holds.find(key); it != holds.end()) return it->second;
    bool result = true;
    side_t Bs[144], Ms[144];
    int id[144], n = 0;
    for (side_t ms = side_mask & ~(m|b); ms && result; ms &= ms-1) {
      const side_t mp = m | (ms & -ms);
      if (won(mp)) {
        result = false;
        break;
      }
      for (int r = 0; r < 8 && result; r++) {
        const side_t m2 = T.rot(mp, r >> 1, r & 1), b2 = T.rot(b, r >> 1, r & 1);
        if (won(b2)) continue;  // Win or tie for b
        if (won(m2)) {
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
    holds[key] = result;
    return result;
  }
};

tie_prover_t::tie_prover_t(const int memo_bits)
  : impl(new impl_t(memo_bits)) {}

tie_prover_t::~tie_prover_t() {}

tie_result_t tie_prover_t::prove(const high_board_t board, const int max_depth, const double seconds) {
  GEODE_ASSERT(!board.middle() && !board.done() && max_depth >= 0);
  impl_t& I = *impl;
  I.new_board(seconds);
  const side_t a = board.side(board.turn()), o = board.side(!board.turn());
  tie_result_t result;
  for (int d = 0; d <= max_depth && !I.timeout && (result.mover < 0 || result.other < 0); d++) {
    if (result.mover < 0 && I.mover_holds(a, o, d)) result.mover = d;
    if (!I.timeout && result.other < 0 && I.other_holds(a, o, d)) result.other = d;
  }
  result.timeout = I.timeout;
  return result;
}

void tie_prover_t::set_time_check_bits(const int bits) {
  GEODE_ASSERT(0 <= bits && bits < 63);
  impl->time_check_mask = (int64_t(1) << bits) - 1;
}

void tie_prover_t::set_paranoid(const bool paranoid) {
  impl->paranoid = paranoid;
}

bool tie_prover_t::undo_holds(const high_board_t board) {
  GEODE_ASSERT(!board.middle() && !board.done());
  impl_t& I = *impl;
  I.new_board(1e9);
  return I.breaker_holds(board.side(board.turn()), board.side(!board.turn()));
}

}  // namespace pentago
