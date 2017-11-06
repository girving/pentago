// Board enumeration

#include "pentago/base/section.h"
#include "pentago/base/symmetry.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/random.h"
#include "pentago/utility/str.h"
namespace pentago {

using std::get;
using std::make_tuple;
using std::swap;
using std::tuple;
using std::vector;

section_t count(board_t board) {
  return section_t(vec(count(quadrant(board,0)),count(quadrant(board,1)),count(quadrant(board,2)),count(quadrant(board,3))));
}

tuple<RawArray<const quadrant_t>,int> rotation_minimal_quadrants(int black, int white) {
  GEODE_ASSERT(0<=black && 0<=white && black+white<=9);
  const int i = ((black*(21-black))>>1)+white;
  const uint16_t lo = rotation_minimal_quadrants_offsets[i],
                 hi = rotation_minimal_quadrants_offsets[i+1];
  const int moved = rotation_minimal_quadrants_reflect_moved[i];
  return make_tuple(RawArray<const quadrant_t>(hi-lo,rotation_minimal_quadrants_flat+lo),moved);
}

tuple<RawArray<const quadrant_t>,int> rotation_minimal_quadrants(Vector<uint8_t,2> counts) {
  return rotation_minimal_quadrants(counts[0],counts[1]);
}

RawArray<const quadrant_t> safe_rmin_slice(Vector<uint8_t,2> counts, int lo, int hi) {
  RawArray<const quadrant_t> rmin = get<0>(rotation_minimal_quadrants(counts));
  GEODE_ASSERT(0<=lo && lo<=hi && (unsigned)hi<=(unsigned)rmin.size());
  return rmin.slice(lo,hi);
}

RawArray<const quadrant_t> safe_rmin_slice(Vector<uint8_t,2> counts, Range<int> range) {
  return safe_rmin_slice(counts,range.lo,range.hi);
}

Vector<int,4> section_t::shape() const {
  return vec(get<0>(rotation_minimal_quadrants(counts[0])).size(),
             get<0>(rotation_minimal_quadrants(counts[1])).size(),
             get<0>(rotation_minimal_quadrants(counts[2])).size(),
             get<0>(rotation_minimal_quadrants(counts[3])).size());
}

uint64_t section_t::size() const {
  const auto s = shape();
  return (uint64_t)s[0]*s[1]*s[2]*s[3];
}

section_t section_t::child(int quadrant) const {
  GEODE_ASSERT((unsigned)quadrant<4 && counts[quadrant].sum()<9);
  section_t child = *this;
  child.counts[quadrant][sum()&1]++;
  return child;
}

section_t section_t::parent(int quadrant) const {
  GEODE_ASSERT((unsigned)quadrant<4);
  section_t parent = *this;
  auto& count = parent.counts[quadrant][!(sum()&1)];
  GEODE_ASSERT(count);
  count--;
  GEODE_ASSERT(parent.child(quadrant)==*this);
  return parent;
}

section_t section_t::transform(uint8_t global) const {
  const int r = global&3;
  static uint8_t source[4][4] = {{0,1,2,3},{1,3,0,2},{3,2,1,0},{2,0,3,1}};
  section_t t(vec(counts[source[r][0]],counts[source[r][1]],counts[source[r][2]],counts[source[r][3]]));
  if (global&4)
    swap(t.counts[0],t.counts[3]);
  return t;
}

template<int symmetries> tuple<section_t,uint8_t> section_t::standardize() const {
  static_assert(symmetries==1 || symmetries==4 || symmetries==8,"");
  section_t best = *this;
  uint8_t best_g = 0;
  for (int g=1;g<symmetries;g++) {
    section_t t = transform(g);
    if (best > t) {
      best = t;
      best_g = g;
    }
  }
  return make_tuple(best,best_g);
}

template tuple<section_t,uint8_t> section_t::standardize<4>() const;
template tuple<section_t,uint8_t> section_t::standardize<8>() const;

Vector<int,4> section_t::quadrant_permutation(uint8_t symmetry) {
  GEODE_ASSERT(symmetry<8);
  typedef Vector<uint8_t,2> CV;
  section_t s = section_t(vec(CV(0,0),CV(1,0),CV(2,0),CV(3,0))).transform(symmetry);
  Vector<int,4> p;
  for (int i=0;i<4;i++)
    p[i] = s.counts.find(CV(i,0));
  return p;
}

// For python exposure
section_t standardize_section(section_t s, int symmetries) {
  GEODE_ASSERT(symmetries==1 || symmetries==4 || symmetries==8);
  return symmetries==1 ? s : symmetries==4 ? get<0>(s.standardize<4>()) : get<0>(s.standardize<8>());
}

bool section_t::valid() const {
  for (int i=0;i<4;i++)
    if ((int)counts[i][0]+counts[i][1] > 9)
      return false;
  const int black = counts[0][0]+counts[1][0]+counts[2][0]+counts[3][0],
            white = counts[0][1]+counts[1][1]+counts[2][1]+counts[3][1];
  if (black<white || black>white+1)
    return false;
  return true;
}

bool section_valid(Vector<Vector<uint8_t,2>,4> s) {
  return section_t(s).valid();
}

ostream& operator<<(ostream& output, section_t section) {
  output << section.sum() << '-';
  for (int i=0;i<4;i++)
    for (int j=0;j<2;j++)
      output << (int)section.counts[i][j];
  return output;
}

board_t random_board(Random& random, const section_t& section) {
  board_t board = 0;
  int permutation[9] = {0,1,2,3,4,5,6,7,8};
  for (int q=0;q<4;q++) {
    for (int i=0;i<8;i++)
      swap(permutation[i],permutation[random.uniform<int>(i,9)]);
    quadrant_t side0 = 0, side1 = 0;
    int b = section.counts[q][0],
        w = section.counts[q][1];
    for (int i=0;i<b;i++)
      side0 |= 1<<permutation[i];
    for (int i=b;i<b+w;i++)
      side1 |= 1<<permutation[i];
    board |= (uint64_t)pack(side0,side1)<<16*q;
  }
  return board;
}

tuple<quadrant_t,uint8_t> rotation_standardize_quadrant(quadrant_t q) {
  quadrant_t minq = q;
  uint8_t minr = 0;
  side_t s[4][2];
  s[0][0] = unpack(q,0);
  s[0][1] = unpack(q,1);
  for (int r=0;r<3;r++) {
    for (int i=0;i<2;i++)
      s[r+1][i] = rotations[s[r][i]][0];
    quadrant_t rq = pack(s[r+1][0],s[r+1][1]);
    if (minq > rq) {
      minq = rq;
      minr = r+1;
    }
  }
  return make_tuple(minq,minr);
}

static string show_quadrant_rmins(const quadrant_t quadrant) {
  const auto c = count(quadrant);
  const auto rmin = rotation_minimal_quadrants(c);
  const int ir = rotation_minimal_quadrants_inverse[quadrant];
  return format("%d%d-%03d-%d%c",c[0],c[1],ir/4,ir&3,ir/4<get<1>(rmin)?"ur"[(ir/4)&1]:'i');
}

string show_board_rmins(const board_t board) {
  return format("%s %s\n%s %s",show_quadrant_rmins(quadrant(board,1)),
                               show_quadrant_rmins(quadrant(board,3)),
                               show_quadrant_rmins(quadrant(board,0)),
                               show_quadrant_rmins(quadrant(board,2)));
}

}
