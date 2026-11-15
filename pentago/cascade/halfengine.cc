// Rotation abstracted forward search over halfsuper_t

#include "pentago/cascade/halfengine.h"
#include "pentago/cascade/tie.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/popcount.h"
#include "pentago/utility/wasm_alloc.h"
#include <cstring>
#ifndef __wasm__
#include <cstdlib>
#endif
namespace pentago {

#ifndef __wasm__
using std::free;
using std::malloc;
#endif

// Functions rather than globals, since freestanding wasm runs no static constructors
static inline halfsuper_t none() { return halfsuper_t(0, 0); }
static inline halfsuper_t all() { return halfsuper_t(~uint64_t(0), ~uint64_t(0)); }

// Arbitrarily rotated win distances, as in pentago/base/score.cc and precompute.cc
namespace {
struct distances_t {
  uint16_t pack3[512];             // 9 bit quadrant -> radix 3 digits (black 1, white 2)
  uint64_t table[4][19683][2];     // [quadrant][radix 3 quadrant][word]: 32 4-bit distances
  uint32_t cover[4][512];          // Lines whose segment in quadrant q some rotation of q covers
  uint64_t covers[4][512][2];      // 4 bit masks of the rotations of q covering each line's segment
  uint8_t line_quads[32][4];       // Quadrants each line passes through, then 4 for padding
  halfsuper_t only[4][2][16];      // [quadrant][parity][rotation mask]: states rotating q by the mask
  uint16_t rot[4][512];            // Quadrant q as it stands after rotation r of a halfsuper_t

  WASM_SMALL void init() {
    // The 32 five in a row lines
    side_t lines[32];
    int n = 0;
    const auto bit = [](const int x, const int y) {
      return side_t(1) << (16*(2*(x/3) + y/3) + 3*(x%3) + y%3);
    };
    const int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
    for (const auto& d : dirs)
      for (int x = 0; x < 6; x++)
        for (int y = 0; y < 6; y++) {
          const int ex = x + 4*d[0], ey = y + 4*d[1];
          if (ex < 0 || ex >= 6 || ey < 0 || ey >= 6) continue;
          side_t line = 0;
          for (int i = 0; i < 5; i++) line |= bit(x + i*d[0], y + i*d[1]);
          lines[n++] = line;
        }
    NON_WASM_ASSERT(n == 32);

    // Quadrant rotations: rot[r][q] is q rotated by r quarter turns
    uint16_t rot[4][512];
    for (int q = 0; q < 512; q++) {
      rot[0][q] = q;
      int s = 0;
      for (int i = 0; i < 9; i++)
        if (q >> i & 1) s |= 1 << (3*(i%3) + 2 - i/3);
      rot[1][q] = s;
    }
    for (int r = 2; r < 4; r++)
      for (int q = 0; q < 512; q++) rot[r][q] = rot[1][rot[r-1][q]];

    // Radix 3 packing
    for (int q = 0; q < 512; q++) {
      int p = 0;
      for (int i = 8; i >= 0; i--) p = 3*p + (q >> i & 1);
      pack3[q] = p;
    }

    // Coverage tables.  Rotation r of a quadrant in a halfsuper_t moves its stones by rot[-r].
    memset(covers, 0, sizeof(covers));
    for (int q = 0; q < 4; q++)
      for (int b = 0; b < 512; b++) {
        uint32_t c = 0;
        for (int i = 0; i < 32; i++) {
          const int qp = lines[i] >> 16*q & 0x1ff;
          for (int r = 0; r < 4; r++)
            if (!(qp & ~rot[(4-r)&3][b])) {
              c |= uint32_t(1) << i;
              covers[q][b][i/16] |= uint64_t(1) << (4*(i%16) + r);
            }
        }
        cover[q][b] = c;
      }
    for (int i = 0; i < 32; i++) {
      int k = 0;
      for (int q = 0; q < 4; q++)
        if (lines[i] >> 16*q & 0x1ff) line_quads[i][k++] = q;
      for (; k < 4; k++) line_quads[i][k] = 4;
    }

    // Rotation r of a quadrant in a halfsuper_t moves its stones by rot[-r]
    for (int r = 0; r < 4; r++)
      for (int q = 0; q < 512; q++) this->rot[r][q] = rot[(4-r)&3][q];

    // Halfsuper index i holds rotations (2a + (b+c+d+p)%2, b, c, d) for i = a + 2b + 8c + 32d
    for (int q = 0; q < 4; q++)
      for (int p = 0; p < 2; p++)
        for (int m = 0; m < 16; m++) {
          halfsuper_t h(0, 0);
          for (int i = 0; i < 128; i++) {
            const int a = i&1, b = i>>1&3, c = i>>3&3, d = i>>5&3;
            const int r[4] = {2*a + (b+c+d+p)%2, b, c, d};
            if (m >> r[q] & 1) h |= halfsuper_t::singleton(i);
          }
          only[q][p][m] = h;
        }

    // Distance table: moves black needs within each quadrant, or 4 if white blocks the line
    memset(table, 0, sizeof(table));
    for (int b = 0; b < 512; b++)
      for (int w = 0; w < 512; w++) {
        if (b & w) continue;
        const int index = pack3[b] + 2*pack3[w];
        for (int q = 0; q < 4; q++)
          for (int i = 0; i < 32; i++) {
            const int qp = lines[i] >> 16*q & 0x1ff;
            int d = 4;
            for (int r = 0; r < 4; r++)
              if (!(rot[r][w] & qp)) {
                const int c = popcount(uint64_t(qp & ~rot[r][b]));
                if (d > c) d = c;
              }
            table[q][index][i/16] |= uint64_t(d) << 4*(i%16);
          }
      }
  }
};
}  // namespace

static distances_t* distances = 0;

// Build the tables on first use, out of line since that happens once
WASM_SMALL __attribute__((noinline)) static void build_distances() {
  distances = static_cast<distances_t*>(malloc(sizeof(distances_t)));
  distances->init();
}

static inline const distances_t& get_distances() {
  if (__builtin_expect(!distances, 0)) build_distances();
  return *distances;
}

// Same as halfsuper_wins, but faster: a line contributes only if every quadrant it passes through
// can cover its segment under some rotation, and few lines are that alive
template<int parities> static inline void side_wins_helper(const side_t side, const bool parity,
                                                           halfsuper_t* wins) {
  const auto& D = *distances;
  const int q0 = quadrant(side, 0), q1 = quadrant(side, 1), q2 = quadrant(side, 2), q3 = quadrant(side, 3);
  uint32_t alive = D.cover[0][q0] & D.cover[1][q1] & D.cover[2][q2] & D.cover[3][q3];
  for (int p = 0; p < parities; p++) wins[p] = halfsuper_t(0, 0);
  const uint64_t* covers[4] = {D.covers[0][q0], D.covers[1][q1], D.covers[2][q2], D.covers[3][q3]};
  while (alive) {
    const int i = __builtin_ctz(alive);
    alive &= alive - 1;
    const auto& lq = D.line_quads[i];
    const int shift = 4*(i&15), w = i>>4;
    const int m0 = covers[lq[0]][w] >> shift & 15, m1 = covers[lq[1]][w] >> shift & 15;
    const int m2 = lq[2] < 4 ? covers[lq[2]][w] >> shift & 15 : 0;
    for (int p = 0; p < parities; p++) {
      const bool par = parities == 1 ? parity : p;
      halfsuper_t h = D.only[lq[0]][par][m0] & D.only[lq[1]][par][m1];
      if (lq[2] < 4) h &= D.only[lq[2]][par][m2];
      wins[p] |= h;
    }
  }
}

// Same as halfsuper_wins, but faster: a line contributes only if every quadrant it passes through
// can cover its segment under some rotation, and few lines are that alive
static inline halfsuper_t side_wins(const side_t side, const bool parity) {
  halfsuper_t w;
  side_wins_helper<1>(side, parity, &w);
  return w;
}

halfsuper_t halfengine_t::fast_wins(const side_t side, const bool parity) {
  get_distances();
  return side_wins(side, parity);
}

side_t halfengine_t::rotate_state(const side_t side, const bool parity, const int i) {
  const auto& D = get_distances();
  const int a = i&1, b = i>>1&3, c = i>>3&3, d = i>>5&3;
  const int r[4] = {2*a + (b+c+d+parity)%2, b, c, d};
  side_t s = 0;
  for (int q = 0; q < 4; q++) s |= side_t(D.rot[r[q]][quadrant(side, q)]) << 16*q;
  return s;
}

int halfengine_t::closeness(const side_t black, const side_t white) {
  const auto& D = get_distances();
  int q[4];
  for (int i = 0; i < 4; i++)
    q[i] = D.pack3[quadrant(black, i)] + 2*D.pack3[quadrant(white, i)];
  const uint64_t each = 0x1111111111111111;
  uint64_t d[2];
  for (int i = 0; i < 2; i++) {
    const uint64_t d0 = D.table[0][q[0]][i], d1 = D.table[1][q[1]][i],
                   d2 = D.table[2][q[2]][i], d3 = D.table[3][q[3]][i],
                   blocks = (d0|d1|d2|d3) & 4*each,
                   blocked = blocks | blocks>>1 | blocks>>2,
                   unblocked = ~blocked;
    d[i] = (d0&unblocked) + (d1&unblocked) + (d2&unblocked) + (d3&unblocked) + (6*each&blocked);
  }
  // Fieldwise minimum of 4 bit fields
  const auto fmin = [=](const uint64_t a, const uint64_t b) {
    uint64_t mask = ((each<<3 | a) - b) & each<<3;
    mask |= mask>>1;
    mask |= mask>>2;
    return (a&~mask) | (b&mask);
  };
  uint64_t m = fmin(d[0], d[1]);
  m = fmin(m, m>>32);
  m = fmin(m, m>>16);
  m = fmin(m, m>>8);
  const int min_distance = fmin(m, m>>4) & 0xf;
  if (min_distance == 6) return 0;
  const uint64_t mins = min_distance*each;
  const auto matches = [=](const uint64_t x) { return ~((x^mins) | (x^mins)>>1 | (x^mins)>>2) & each; };
  return ((6 - min_distance) << 16) + popcount(matches(d[0]) | matches(d[1]) << 1);
}

// Transposition table entries: tag = hash >> bits in the high 54 bits, depth in the low 10
struct halfengine_t::entry_t {
  uint64_t key;
  uint64_t known[2], wins[2];
};
static_assert(sizeof(halfengine_t::entry_t) == 40);

struct halfengine_t::node_t {
  uint64_t hash;      // Zero if there is nothing to store
  halfinfo_t info;
  halfsuper_t wins1;  // Wins for side1 at this node's parity
};

static inline halfsuper_t load(const uint64_t x[2]) { return halfsuper_t(x[0], x[1]); }
static inline void save(uint64_t x[2], const halfsuper_t h) { memcpy(x, &h, sizeof(h)); }

static inline int count(const halfsuper_t h) {  // popcount(halfsuper_s) isn't in the wasm build
  uint64_t x[2];
  save(x, h);
  return popcount(x[0]) + popcount(x[1]);
}

static inline uint64_t mix(uint64_t x) {
  x ^= x >> 31;
  x *= 0x7fb5d329728ea185;
  x ^= x >> 27;
  x *= 0x81dadef4bc2dd44d;
  return x ^ x >> 33;
}

// side_t uses bits 0-8, 16-24, 32-40, 48-56, so bits 62 and 63 are free for flags
static inline uint64_t hash_node(const bool aggressive, const bool parity, const side_t side0, const side_t side1) {
  const uint64_t h = mix(mix(side0 | uint64_t(aggressive) << 63 | uint64_t(parity) << 62) ^ side1);
  return h ? h : 1;
}

WASM_SMALL halfengine_t::halfengine_t(const int bits)
  : bits(bits) {
  NON_WASM_ASSERT(10 <= bits && bits <= 30);
  table = static_cast<entry_t*>(malloc(sizeof(entry_t) << bits));
  clear();
  get_distances();  // Build the tables
}

halfengine_t::~halfengine_t() {
  free(table);
}

void halfengine_t::clear() {
  memset(static_cast<void*>(table), 0, sizeof(entry_t) << bits);
}

// Evaluate everything we can about a position without recursing into children
template<bool aggressive> halfengine_t::node_t
halfengine_t::shallow(const int depth, const bool parity, const side_t side0, const side_t side1,
                      const halfsuper_t wins0, const halfsuper_t wins1, const halfsuper_t interesting) {
  node_t node;
  node.hash = 0;
  node.wins1 = wins1;
  halfinfo_t& info = node.info;

  // Check whether the current state is a win for either player
  info.known = wins0 | wins1;
  info.wins = aggressive ? wins0 & ~wins1 : wins0;

  // If zero moves remain, the passive player wins
  if (!depth) {
    if (!aggressive) info.wins |= ~info.known;
    info.known = all();
    return node;
  }

  // If we're passive and only one move remains, any safe move is a win.  The rotation lands in the
  // other parity, so we need side1's wins there.
  if (!aggressive && depth <= 1) {
    const halfsuper_t safe = rmax(~side_wins(side1, !parity));
    info.wins |= safe & ~info.known;
    info.known |= safe;
  }

  // Exit early if possible
  if (!(interesting & ~info.known)) return node;

  // Look up the position in the transposition table.  If the entry is too shallow, only exact
  // information survives: wins when aggressive, losses otherwise.
  node.hash = hash_node(aggressive, parity, side0, side1);
  const entry_t& e = table[node.hash & ((uint64_t(1) << bits) - 1)];
  if (e.key >> 10 == node.hash >> bits) {
    halfsuper_t known = load(e.known), wins = load(e.wins);
    if (depth > int(e.key & 1023)) {
      known &= aggressive ? wins : ~wins;
      wins = aggressive ? known : none();
    }
    // Immediate information is exact, so it overrides the table
    info.wins |= wins & ~info.known;
    info.known |= known;
  }
  return node;
}

// Store a search result, merging with any existing entry for the same position
template<bool aggressive> void halfengine_t::store(const int depth, const uint64_t hash, const halfinfo_t& info) {
  entry_t& e = table[hash & ((uint64_t(1) << bits) - 1)];
  const bool same = e.key >> 10 == hash >> bits;
  const int old_depth = e.key & 1023;
  if (same || depth >= old_depth) {
    halfinfo_t mine = info;
    if (same) {
      halfinfo_t existing{load(e.known), load(e.wins)};
      // Discard low depth information, keeping only exact results
      const int max_depth = depth > old_depth ? depth : old_depth;
      if (depth < max_depth) {
        const halfsuper_t m = aggressive ? mine.wins : ~mine.wins;
        mine.known &= m;
        mine.wins &= m;
      }
      if (old_depth < max_depth) {
        const halfsuper_t m = aggressive ? existing.wins : ~existing.wins;
        existing.known &= m;
        existing.wins &= m;
      }
      // Disagreements are between exact information and depth limited information of the
      // opposite kind, both true (see pentago/search/supertable.cc).  Keep the exact value.
      const halfsuper_t conflict = (mine.wins ^ existing.wins) & mine.known & existing.known;
      if (conflict) {
        const halfsuper_t exact = aggressive ? conflict : none();
        mine.wins = (mine.wins & ~conflict) | exact;
        existing.wins = (existing.wins & ~conflict) | exact;
      }
      existing.known |= mine.known;
      existing.wins = existing.wins ^ ((existing.wins ^ mine.wins) & mine.known);
      mine = existing;
      e.key = (hash >> bits) << 10 | max_depth;
    } else {
      e.key = (hash >> bits) << 10 | depth;
    }
    save(e.known, mine.known);
    save(e.wins, mine.wins);
  }
}

// Evaluate a position for all important rotations, returning the set of rotations in which we win.
// The returned set of known rotations includes all important rotations, but may include others.
template<bool aggressive> halfinfo_t
halfengine_t::recurse(const int depth, const bool parity, const side_t side0, const side_t side1,
                      node_t node, const halfsuper_t important) {
  halfinfo_t& info = node.info;
  halfsuper_t possible = none();  // Keep track of possible wins that we might have missed

  // Tie prover cutoffs
  if (cutoffs.prover && depth >= cutoffs.min_depth) {
    const halfsuper_t todo = important & ~info.known;
    if (count(todo) <= cutoffs.max_states) {
      for (int i = 0; i < 128; i++) {
        if (!todo[i]) continue;
        const side_t a = rotate_state(side0, parity, i), b = rotate_state(side1, parity, i);
        const halfsuper_t bit = halfsuper_t::singleton(i);
        cutoff_calls++;
        if (aggressive ? cutoffs.prover->other_holds(a, b, cutoffs.depth, cutoffs.seconds)
                       : cutoffs.prover->mover_holds(a, b, cutoffs.depth, cutoffs.seconds)) {
          cutoff_hits++;
          info.known |= bit;
          if (!aggressive) info.wins |= bit;
        }
      }
      if (!(important & ~info.known)) {
        if (node.hash) store<aggressive>(depth, node.hash, info);
        return info;
      }
    }
  }

  // Collect possible moves.  Each is side0 with one more stone.
  side_t free_ = side_mask ^ (side0 | side1);
  int total = 0;
  side_t moves[36];
  while (free_) {
    const side_t m = free_ & -free_;
    moves[total++] = side0 | m;
    free_ ^= m;
  }

  // Do a shallow evaluation of each move.  Children have the other parity.
  const halfsuper_t theirs = node.wins1, theirs_child = side_wins(side1, !parity);
  node_t children[36];
  if (depth > 1)  // Prefetch children's table entries
    for (int i = 0; i < total; i++)
      __builtin_prefetch(&table[hash_node(!aggressive, !parity, side1, moves[i]) & ((uint64_t(1) << bits) - 1)]);
  for (int i = total - 1; i >= 0; i--) {
    const side_t move = moves[i];
    // Do we win without a rotation?  If we're passive, it's safe to wait until after the rotation.
    halfsuper_t ours[2];  // Indexed by parity
    if (aggressive) {
      side_wins_helper<2>(move, parity, ours);
      const halfsuper_t immediate = ours[parity] & ~theirs;
      info.wins |= immediate & ~info.known;
      info.known |= immediate;
    } else
      ours[!parity] = side_wins(move, !parity);

    // Do a shallow evaluation of the child position
    const halfsuper_t mask = rmax(important & ~info.known);
    children[i] = shallow<!aggressive>(depth - 1, !parity, side1, move, theirs_child, ours[!parity], mask);
    const halfinfo_t& child = children[i].info;
    const halfsuper_t wins = rmax(~child.wins & child.known);
    info.wins |= wins & ~info.known;
    info.known |= wins;

    // Exit early if possible
    if (!(important & ~info.known)) goto done;

    // Discard the move if we already know enough about it
    if (!(mask & ~child.known)) {
      possible |= rmax(~child.known);
      moves[i] = moves[total-1];
      children[i] = children[total-1];
      total--;
    }
  }

  {
    // Check how close we are to an aggressive win, pretending the aggressor may rotate arbitrarily
    int order[36];  // We'll sort moves in ascending order of this array
    for (int i = total - 1; i >= 0; i--) {
      const int closeness = aggressive ? halfengine_t::closeness(moves[i], side1)
                                       : halfengine_t::closeness(side1, moves[i]);
      const int distance = 6 - (closeness >> 16);
      if (distance == 6 || distance > (depth - aggressive)/2) {  // No aggressive win within depth
        if (!aggressive) {
          info.wins = info.known = all();
          goto done;
        } else {
          moves[i] = moves[total-1];
          children[i] = children[total-1];
          order[i] = order[total-1];
          total--;
          continue;
        }
      }
      order[i] = aggressive ? -closeness : closeness;
    }

    // Are we out of recursion depth?
    if (depth == 1 && total) goto done;

    // Sort moves and children by order
    for (int i = 1; i < total; i++)
      for (int j = i; j > 0 && order[j-1] > order[j]; j--) {
        const int o = order[j]; order[j] = order[j-1]; order[j-1] = o;
        const side_t m = moves[j]; moves[j] = moves[j-1]; moves[j-1] = m;
        const node_t c = children[j]; children[j] = children[j-1]; children[j-1] = c;
      }

    // Recurse
    for (int i = 0; i < total; i++) {
      const halfsuper_t mask = rmax(important & ~info.known);
      const halfinfo_t child = recurse<!aggressive>(depth - 1, !parity, side1, moves[i], children[i], mask);
      const halfsuper_t wins = rmax(~child.wins & child.known);
      info.wins |= wins & ~info.known;
      info.known |= wins;
      if (!(important & ~info.known)) goto done;
      possible |= rmax(~child.known);
    }

    // If we've analyzed all children, any move that isn't a possible win is a loss
    info.known |= ~possible;
  }

  done:
  if (node.hash) store<aggressive>(depth, node.hash, info);
  return info;
}

WASM_SMALL bool halfengine_t::evaluate(const bool aggressive, int depth, const side_t side0, const side_t side1) {
  NON_WASM_ASSERT(depth >= 0 && !(side0 & side1));
  const int empty = 36 - popcount(side0 | side1);
  if (depth > empty) depth = empty;
  const halfsuper_t root = halfsuper_t::singleton(0);
  const halfsuper_t w0 = side_wins(side0, 0), w1 = side_wins(side1, 0);
  halfinfo_t info;
  if (aggressive) {
    const node_t node = shallow<true>(depth, 0, side0, side1, w0, w1, root);
    info = node.info.known[0] || !depth ? node.info : recurse<true>(depth, 0, side0, side1, node, root);
  } else {
    const node_t node = shallow<false>(depth, 0, side0, side1, w0, w1, root);
    info = node.info.known[0] || !depth ? node.info : recurse<false>(depth, 0, side0, side1, node, root);
  }
  NON_WASM_ASSERT(info.known[0]);
  return info.wins[0];
}

WASM_SMALL int halfengine_t::search(const side_t side0, const side_t side1, const double seconds, double (*clock)()) {
  const int full = 36 - popcount(side0 | side1);
  const double start = clock();
  for (int d = 1; d <= full; d++) {
    bool not_win = false;
    if (d & 1 || d == full) {
      if (evaluate(true, d, side0, side1)) return 1;
      not_win = true;
    }
    if (!(d & 1) || d == full) {
      if (!evaluate(false, d, side0, side1)) return -1;
      if (d == full && not_win) return 0;
    }
    if (clock() - start > seconds) break;
  }
  return 2;
}

}  // namespace pentago
