// Operations involving pentago symmetries

#include <pentago/symmetry.h>
#include <other/core/array/Array.h>
#include <other/core/array/NdArray.h>
#include <other/core/math/integer_log.h>
#include <other/core/random/Random.h>
#include <other/core/structure/Hashtable.h>
#include <other/core/python/module.h>
#include <other/core/utility/interrupts.h>
namespace pentago {

using std::cout;
using std::endl;

const symmetries_t symmetries;

ostream& operator<<(ostream& output, symmetry_t s) {
  return output<<format("(%d,%d,%d=%d%d%d%d)",s.global>>2,s.global&3,s.local,s.local&3,s.local>>2&3,s.local>>4&3,s.local>>6);
}

ostream& operator<<(ostream& output, local_symmetry_t s) {
  return output<<format("(%d=%d%d%d%d)",s.local,s.local&3,s.local>>2&3,s.local>>4&3,s.local>>6);
}

static void group_test() {
  // Test identity and inverses
  const symmetry_t e = 0;
  for (auto g : symmetries) {
    OTHER_ASSERT(g*e==g);
    OTHER_ASSERT(e*g==g);
    OTHER_ASSERT(g*g.inverse()==e);
  }

  // Test cancellativitiy
  for (auto a : symmetries)
    for (auto b : symmetries) {
      OTHER_ASSERT((b*a.inverse())*a==b);
      OTHER_ASSERT(a*(a.inverse()*b)==b);
    }

  // Test associativity following Rajagopalany and Schulman, Verification of identities, 1999.
  // First, find a generating subset.
  Array<symmetry_t> generators;
  Array<symmetry_t> generated;
  generated.preallocate(symmetries.size());
  Hashtable<symmetry_t> generated_set;
  generated.append(e);
  generated_set.set(e);
  for (auto g : symmetries)
    if (!generated_set.contains(g)) {
      generators.append(g);
      for (;;) {
        RawArray<const symmetry_t> previous = generated;
        for (auto a : previous)
          if (generated_set.set(g*a))
            generated.append_assuming_enough_space(g*a);
        if (previous.size()==generated.size())
          break;
      }
    }
  OTHER_ASSERT(generated.size()==symmetries.size());
  OTHER_ASSERT(generators.size()<=integer_log(symmetries.size())+2);
  // Check associativity on the generators.  This is sufficient thanks to cancellativity.
  for (auto a : generators)
    for (auto b : generators)
      for (auto c : generators)
        OTHER_ASSERT((a*b)*c==a*(b*c));

  // Check products of local with general
  for (auto a : symmetries)
    for (int r : range(256)) {
      local_symmetry_t b(r);
      symmetry_t sb(b);
      OTHER_ASSERT(a*b==a*sb);
      OTHER_ASSERT(b*a==sb*a);
    }
}

// rotate_quadrants[r][q] is the quadrant moved to q under rotation r
static const uint8_t rotate_quadrants[4][4] = {{0,1,2,3},{1,3,0,2},{3,2,1,0},{2,0,3,1}};

side_t transform_side(symmetry_t s, side_t side) {
  // Decompose into quadrants
  quadrant_t q[4] = {quadrant(side,0),quadrant(side,1),quadrant(side,2),quadrant(side,3)};
  // Apply local rotations plus global rotations confined to each quadrant
  for (int i=0;i<4;i++)
    switch ((s.global+(s.local>>2*i))&3) {
      case 1: q[i] = rotations[q[i]][0]; break;
      case 2: q[i] = rotations[rotations[q[i]][0]][0]; break;
      case 3: q[i] = rotations[q[i]][1]; break;
    }
  // Move quadrants around according to global rotation
  const int g = s.global&3;
  quadrant_t qr[4] = {q[rotate_quadrants[g][0]],q[rotate_quadrants[g][1]],q[rotate_quadrants[g][2]],q[rotate_quadrants[g][3]]};
  // Reflect if necessary
  if (s.global&4) {
    swap(qr[0],qr[3]);
    for (int i=0;i<4;i++)
      qr[i] = reflections[qr[i]];
  }
  return quadrants(qr[0],qr[1],qr[2],qr[3]);
}

// I don't trust the compiler to figure out that all the branching is shared between sides,
// so this routine is two manually inlined copies of transform_side
board_t transform_board(symmetry_t s, board_t board) {
  // Decompose into quadrants
  const quadrant_t qp[4] = {quadrant(board,0),quadrant(board,1),quadrant(board,2),quadrant(board,3)};
  quadrant_t q[4][2] = {{unpack(qp[0],0),unpack(qp[0],1)},{unpack(qp[1],0),unpack(qp[1],1)},{unpack(qp[2],0),unpack(qp[2],1)},{unpack(qp[3],0),unpack(qp[3],1)}};
  // Apply local rotations plus global rotations confined to each quadrant
  for (int i=0;i<4;i++)
    switch ((s.global+(s.local>>2*i))&3) {
      case 1: q[i][0] = rotations[q[i][0]][0];
              q[i][1] = rotations[q[i][1]][0]; break;
      case 2: q[i][0] = rotations[rotations[q[i][0]][0]][0];
              q[i][1] = rotations[rotations[q[i][1]][0]][0]; break;
      case 3: q[i][0] = rotations[q[i][0]][1];
              q[i][1] = rotations[q[i][1]][1]; break;
    }
  // Move quadrants around according to global rotation
  const int g = s.global&3;
  quadrant_t qr[4][2] = {{q[rotate_quadrants[g][0]][0],q[rotate_quadrants[g][0]][1]},
                         {q[rotate_quadrants[g][1]][0],q[rotate_quadrants[g][1]][1]},
                         {q[rotate_quadrants[g][2]][0],q[rotate_quadrants[g][2]][1]},
                         {q[rotate_quadrants[g][3]][0],q[rotate_quadrants[g][3]][1]}};
  // Reflect if necessary
  if (s.global&4) {
    swap(qr[0][0],qr[3][0]);
    swap(qr[0][1],qr[3][1]);
    for (int i=0;i<4;i++) {
      qr[i][0] = reflections[qr[i][0]];
      qr[i][1] = reflections[qr[i][1]];
    }
  }
  return quadrants(pack(qr[0][0],qr[0][1]),pack(qr[1][0],qr[1][1]),pack(qr[2][0],qr[2][1]),pack(qr[3][0],qr[3][1]));
}

symmetry_t random_symmetry(Random& random) {
  const int g = random.uniform<int>(0,symmetries.size());
  return symmetry_t(g>>8,g&255);
}

static void action_test(int steps) {
  Ref<Random> random = new_<Random>(875431);
  for (int step=0;step<steps;step++) {
    // Generate two random symmetries and sides
    const symmetry_t s0 = random_symmetry(random), s1 = random_symmetry(random);
    const side_t side0 = random_side(random), side1 = random_side(random)&~side0;
    // Check action consistency
    OTHER_ASSERT(transform_side(s0,transform_side(s1,side0))==transform_side(s0*s1,side0));
    OTHER_ASSERT(transform_board(s0,pack(side0,side1))==pack(transform_side(s0,side0),transform_side(s0,side1)));
  }
}

Tuple<board_t,symmetry_t> superstandardize(board_t board) {
  return superstandardize(unpack(board,0),unpack(board,1));
}

static NdArray<board_t> superstandardize_py(NdArray<const board_t> boards) {
  NdArray<board_t> standard(boards.shape,false);
  for (int b=0;b<boards.flat.size();b++)
    standard.flat[b] = superstandardize(boards.flat[b]).x;
  return standard;
}

Tuple<board_t,symmetry_t> superstandardize(side_t side0, side_t side1) {
  // Decompose into quadrants, and compute all rotated and reflected versions
  quadrant_t versions[4][8]; // indexed by quadrant, reflection, rotation
  for (int i=0;i<4;i++) {
    quadrant_t s[2][8]; // indexed by side, reflection, rotation
    s[0][0] = quadrant(side0,i);
    s[1][0] = quadrant(side1,i);
    for (int j=0;j<2;j++) {
      s[j][1] = rotations[s[j][0]][0];
      s[j][2] = rotations[s[j][1]][0];
      s[j][3] = rotations[s[j][0]][1];
      for (int k=0;k<4;k++)
        s[j][4+k] = reflections[s[j][k]];
    }
    for (int j=0;j<8;j++)
      versions[i][j] = pack(s[0][j],s[1][j]);
  }
  // Find the minimum version of each quadrant, both with and without reflection
  quadrant_t qmin[4][2];
  for (int i=0;i<4;i++) {
    qmin[i][0] = min(versions[i][0],versions[i][1],versions[i][2],versions[i][3]);
    qmin[i][1] = min(versions[i][4],versions[i][5],versions[i][6],versions[i][7]);
  }
  // Choose global reflection and rotation via lookup table
  #define PLACE(i,r) ((qmin[i][r]>qmin[(i+1)&3][r])+(qmin[i][r]>qmin[(i+2)&3][r])+(qmin[i][r]>qmin[(i+3)&3][r]))
  #define SIGNATURE(r) (PLACE(3*r,r)+4*(PLACE(1,r)+4*(PLACE(2,r)+4*PLACE(3-3*r,r))))
  const int r0 =    superstandardize_table[SIGNATURE(0)],
            r1 = 3&-superstandardize_table[SIGNATURE(1)];
  const board_t b0 = quadrants(qmin[rotate_quadrants[r0][0]][0],qmin[rotate_quadrants[r0][1]][0],qmin[rotate_quadrants[r0][2]][0],qmin[rotate_quadrants[r0][3]][0]),
                b1 = quadrants(qmin[rotate_quadrants[r1][3]][1],qmin[rotate_quadrants[r1][1]][1],qmin[rotate_quadrants[r1][2]][1],qmin[rotate_quadrants[r1][0]][1]),
                bmin = min(b0,b1);
  const bool reflect = b1<b0;
  const int gr = reflect?r1:r0;
  // Determine local rotations
  uint8_t local[4];
  const int o=4*reflect;
  for (int i=0;i<4;i++) {
    const quadrant_t qm = qmin[i][reflect];
    local[i] = ((qm==versions[i][o+0]?0:qm==versions[i][o+1]?1:qm==versions[i][o+2]?2:3)-gr)&3;
  }
  return tuple(bmin,symmetry_t(reflect<<2|gr,local[0]|local[1]<<2|local[2]<<4|local[3]<<6));
}

void superstandardize_test(int steps) {
  Ref<Random> random = new_<Random>(875431);
  for (int step=0;step<steps;step++) {
    // Generate a random board, with possibly duplicated quadrants
    const side_t side0 = random_side(random),
                 side1 = random_side(random)&~side0;
    const board_t pre = pack(side0,side1);
    const int q = random->uniform<int>(0,256);
    const board_t board = transform_board(random_symmetry(random),quadrants(quadrant(pre,q&3),quadrant(pre,q>>2&3),quadrant(pre,q>>4&3),quadrant(pre,q>>6&3)));
    // Standardize
    board_t standard;
    symmetry_t symmetry;
    superstandardize(board).get(standard,symmetry);
    OTHER_ASSERT(transform_board(symmetry,board)==standard);
    // Compare with manual version
    board_t slow = (uint64_t)-1;
    for (auto s : symmetries)
      slow = min(slow,transform_board(s,board));
    OTHER_ASSERT(standard==slow);
    check_interrupts();
  }
}

super_t transform_super(symmetry_t s, super_t C) {
  // We view C as a subset of the local rotation group L: C(r0,r1,r2,r3) iff (rotation of quadrant i by ri) in C.
  // See the header for more details.

  // First apply the local part: C = C local'
  #define APPLY_BOTH(f,a) ({ C.x = f(C.x,(a)); C.y = f(C.y,(a)); })
  #define ROTATE_RIGHT_MOD_4(x,k) ((_mm_srli_epi16(x,(k))&_mm_set1_epi8(0x11*(0xf>>(k))))|(_mm_slli_epi16(x,(4-(k))&3)&_mm_set1_epi8(0x11*((0xf<<(4-(k)))&0xf))))
  #define ROTATE_RIGHT_MOD_16(x,k) (_mm_srli_epi16(x,(k))|_mm_slli_epi16(x,(16-(k))&15))
  #define ROTATE_RIGHT_MOD_64(x,k) (_mm_srli_epi64(x,(k))|_mm_slli_epi64(x,(64-(k))&63))
  APPLY_BOTH(ROTATE_RIGHT_MOD_4,s.local&3);
  APPLY_BOTH(ROTATE_RIGHT_MOD_16,4*(s.local>>2&3));
  APPLY_BOTH(ROTATE_RIGHT_MOD_64,16*(s.local>>4&3));
  if (s.local&1<<6) {
    const int swap = LE_MM_SHUFFLE(2,3,0,1);
    const __m128i sx = _mm_shuffle_epi32(C.x,swap),
                  sy = _mm_shuffle_epi32(C.y,swap);
    const __m128i low = _mm_set_epi64x(0,~(uint64_t)0);
    C = super_t((sx&low)|(sy&~low),(sy&low)|(sx&~low));
  }
  if (s.local&1<<7)
    swap(C.x,C.y);

  // Define quadrant transposition maps acting on the space of local rotations.  Unfortunately, if we want to be
  // as efficient as possible, we need all (4 choose 2) = 6 transpositions.  In particular, while it is possible
  // to represent any permutation as a product of adjacent transpositions only, such representations are often
  // much longer (e.g., (03) = (01)(12)(23)(12)(01)).

  // (01), (12), and (02) are easy, since they act independently on each 64-bit chunk.  Except for the constants,
  // the code follows http://alaska-kamtchatka.blogspot.com/2011/09/4-matrix-transposition.html.
  #define BIT(i) ((uint64_t)1<<(i))
  #define LOW_HALF_TRANSPOSE(x,i,j) ({ \
    const int ii = 1<<2*i, jj = 1<<2*j, kk = 1<<2*(3-i-j), sh = jj-ii; \
    const uint64_t other = 1|BIT(kk)|BIT(2*kk)|BIT(3*kk); \
    auto t = (x^_mm_srli_epi64(x,sh))&_mm_set1_epi64x(other*(BIT(ii)|BIT(3*ii)|BIT(ii+2*jj)|BIT(3*ii+2*jj))); \
    x ^= t^_mm_slli_epi64(t,sh); \
    t = (x^_mm_srli_epi64(x,2*sh))&_mm_set1_epi64x(other*(BIT(2*ii)|BIT(3*ii)|BIT(2*ii+jj)|BIT(3*ii+jj))); \
    x ^= t^_mm_slli_epi64(t,2*sh); })
  #define LOW_TRANSPOSE(i,j) ({ LOW_HALF_TRANSPOSE(C.x,i,j); LOW_HALF_TRANSPOSE(C.y,i,j); })

  // Transposing quadrants 2 and 3 is analogous, but operates on 16 bit chunks instead of single bits, and knits the two __m128i's together
  #define TRANSPOSE_23() ({ \
    const auto a = other::pack<uint32_t>(0xffff0000,0xffff0000,0,0); \
    auto t = (C.x^_mm_srli_si128(C.x,6))&a; \
    C.x ^= t^_mm_slli_si128(t,6); \
    t = (C.y^_mm_srli_si128(C.y,6))&a; \
    C.y ^= t^_mm_slli_si128(t,6); \
    const auto m = other::pack(0,-1,0,-1); \
    t = (C.x^_mm_slli_si128(C.y,4))&m; \
    C.x ^= t; \
    C.y ^= _mm_srli_si128(t,4); })

  // Hmm.  Transpositions (03) and (13) are extremely nasty, since they cross 64-bit and 128-bit chunks and also involve non-byte-aligned
  // shifts, for which there are no available 64-bit crossing instructions.  Therefore, at the cost of longer products of transpositions,
  // we'll confine ourselves to (01),(02),(12),(23) for now.

  // Armed with the above transpositions, we can now form the conjugate s.global C s.global' by expanding s.global into transpositions
  // and an optional per-quadrant reflection.  See `dihedral` for computation of the minimal sequence of transpositions required to build
  // up each element.  First, we conjugate by the quadrant interchange part of the map.
  #define T(i,j) (j==3?3:(j*(j-1)/2+i))
  static const uint8_t offsets[] = {0,0,3,7,10,13,17,18,20};
  static const uint8_t transpositions[] = { // Here r = global rotate left by 90, s = reflect about x = y line
    /* nothing */                // e    = ()
    T(0,1),T(0,2),T(2,3),        // r    = (01)(02)(23)
    T(0,2),T(0,1),T(2,3),T(0,2), // r^2  = (02)(01)(23)(02)
    T(2,3),T(0,2),T(0,1),        // r^3  = (23)(02)(01)
    T(0,2),T(2,3),T(0,2),        // s    = (02)(23)(02)
    T(0,2),T(1,2),T(2,3),T(1,2), // sr   = (02)(12)(23)(12)
    T(1,2),                      // sr^2 = (12)
    T(0,1),T(2,3) };             // sr^3 = (01)(23)
  // Apply transpositions in reverse order
  for (int i=offsets[s.global+1]-1;i>=offsets[s.global];i--)
    switch (transpositions[i]) {
      case T(0,1): LOW_TRANSPOSE(0,1); break;
      case T(0,2): LOW_TRANSPOSE(0,2); break;
      case T(1,2): LOW_TRANSPOSE(1,2); break;
      case T(2,3): TRANSPOSE_23(); break;
    }

  // Finally, if necessary, we conjugate by the quadrant-local reflection map, which amounts to applying the
  // negation isomorphism to each direct product term in Z_4^4.
  if (s.global&4) {
    #define HALF_NEGATE(x) ({ \
      /* Negate quadrant 0 rotations */ \
      auto t = (x^_mm_srli_epi16(x,2))&_mm_set1_epi8(0x22); \
      x ^= t^_mm_slli_epi16(t,2); \
      /* Negate quadrant 1 rotations */ \
      t = (x^_mm_srli_epi16(x,2*4))&_mm_set1_epi16(0xf0); \
      x ^= t^_mm_slli_epi16(t,2*4); \
      /* Negate quadrant 2 rotations */ \
      t = (x^_mm_srli_epi64(x,2*16))&_mm_set1_epi64x(0xffff0000); \
      x ^= t^_mm_slli_epi64(t,2*16); })
    // Negate rotations of the first three quadrants
    HALF_NEGATE(C.x);
    HALF_NEGATE(C.y);
    // Negate quadrant 3 rotations
    auto t = (C.x^C.y)&other::pack(0,0,-1,-1);
    C.x ^= t;
    C.y ^= t;
  }

  // Whew.
  return C;
}

// A meaningless function invariant to global board transformations
bool meaningless(board_t board, uint64_t salt) {
  bool result = 0;
  for (int g=0;g<8;g++)
    result ^= hash(transform_board(symmetry_t(g,0),board)^salt)&1;
  return result;
}

super_t super_meaningless(board_t board, uint64_t salt) {
  super_t result = 0;
  for (int r=0;r<256;r++)
    if (meaningless(transform_board(symmetry_t(0,r),board),salt))
      result |= super_t::singleton(r);
  return result;
}

static void super_action_test(int steps) {
  // First, test that transform_super satisfies our group theoretic definition:
  //   s(C) = gy(C) = {x in L | g'xgy in C} = {gzy'g' | z in C}
  Ref<Random> random = new_<Random>(72831);
  for (int step=0;step<steps;step++) {
    const symmetry_t s = random_symmetry(random);
    const symmetry_t sg(s.global,0), sl(0,s.local);
    const super_t C = random_super(random);
    super_t correct = 0;
    for (int r=0;r<256;r++)
      if (C(r))
        correct |= super_t::singleton((sg*symmetry_t(0,r)*(sl.inverse()*sg.inverse())).local);
    OTHER_ASSERT(correct==transform_super(s,C));
    check_interrupts();
  }

  // Next, test the motivating definition in terms of invariant functions
  for (int step=0;step<steps;step++) {
    const symmetry_t s = random_symmetry(random);
    const side_t side0 = random_side(random),
                 side1 = random_side(random)&~side0;
    const board_t board = pack(side0,side1);
    OTHER_ASSERT(super_meaningless(transform_board(s,board))==transform_super(s,super_meaningless(board)));
    check_interrupts();
  }
}

}
using namespace pentago;

void wrap_symmetry() {
  OTHER_FUNCTION(group_test)
  OTHER_FUNCTION(action_test)
  OTHER_FUNCTION(superstandardize_test)
  OTHER_FUNCTION(super_action_test)
  OTHER_FUNCTION_2(superstandardize,superstandardize_py)
  OTHER_FUNCTION(meaningless)
}
