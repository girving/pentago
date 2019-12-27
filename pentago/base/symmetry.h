// Operations involving pentago symmetries
//
// The symmetry group of rotation abstracted pentago is a semidirect product of the global
// rotation group D_4 and the local quadrant rotation group Z_4^4.  Despite the fact that
// the rest of the rotation abstracted code is prefixed with super, I somehow managed to
// resist calling the class supersymmetry_t.
//
// In the comments below, x' is the inverse of x.
#pragma once

#include "pentago/base/board.h"
#include "pentago/base/superscore.h"
namespace pentago {

using std::tuple;

// A purely local symmetry
struct local_symmetry_t {
  uint8_t local;

  local_symmetry_t()
    : local(0) {}

  explicit local_symmetry_t(uint8_t local) 
    : local(local) {}

  local_symmetry_t operator*(local_symmetry_t s) const {
    return local_symmetry_t((((local&0x33)+(s.local&0x33))&0x33)+(((local&0xcc)+(s.local&0xcc))&0xcc));
  }

  local_symmetry_t inverse() const {
    return local_symmetry_t(local^(local&0x55)<<1);
  }
};

// Global plus local symmetry
struct symmetry_t {
  // A 1 bit reflect field and A 2-bit counterclockwise rotation field for the board as a whole
  unsigned global : 3;

   // 4 2-bit counterclockwise rotation fields, one per quadrant
  unsigned local : 8;

  symmetry_t() {}

  symmetry_t(zero*)
    : global(0), local(0) {}

  symmetry_t(uint8_t global, uint8_t local)
    : global(global), local(local) {}

  symmetry_t(local_symmetry_t s)
    : global(0), local(s.local) {}

  static uint8_t invert_global(uint8_t global) {
    return global^(~global>>2&(global&1))<<1;
  }

  symmetry_t inverse() const {
    uint8_t li = local_symmetry_t(local).inverse().local; // Invert local
    uint8_t gi = invert_global(global); // Invert global
    // Commute local through global
    return symmetry_t(gi,commute_global_local_symmetries[gi][li]);
  }

  explicit operator bool() const {
    return global||local;
  }

  bool operator==(symmetry_t s) const {
    return global==s.global && local==s.local;
  }

  bool operator!=(symmetry_t s) const {
    return !(*this==s);
  }
};

// Group product
inline symmetry_t operator*(symmetry_t a, symmetry_t b) {
  // We seek x = ag al bg bl = a.global a.local b.global b.local
  // Commute local and global transforms: al bg = bg (bg' al bg) = bg l2, x = ag bg l2 bl
  const uint8_t l2 = commute_global_local_symmetries[b.global][a.local];
  // Unit tests to the rescue
  return symmetry_t(((a.global^b.global)&4)|(((a.global^(a.global&b.global>>2)<<1)+b.global)&3),
                    (local_symmetry_t(l2)*local_symmetry_t(b.local)).local);
}

inline symmetry_t operator*(symmetry_t a, local_symmetry_t b) {
  return symmetry_t(a.global,(local_symmetry_t(a.local)*b).local);
}

inline symmetry_t operator*(local_symmetry_t a, symmetry_t b) {
  const uint8_t l2 = commute_global_local_symmetries[b.global][a.local];
  return symmetry_t(b.global,(local_symmetry_t(l2)*local_symmetry_t(b.local)).local);
}

// A symmetry is a function s : X -> X, where X = Z_6^2 is the set of Pentago squares.
// A side_t is a subset A of X, and a board_t is two such subsets.  These functions compute s(A).
// For example, transform_board(symmetry_t(1,0),board) rotates the whole board left by 90 degrees.
side_t transform_side(symmetry_t s, side_t side) __attribute__((const));
board_t transform_board(symmetry_t s, board_t board) __attribute__((const));

// Let B be the set of boards, and A \subset B a subset of boards invariant to global transforms
// (b in A iff g(b) in A for g in D_4).  Define the super operator f : B -> 2^L by
// f(b) = {x in L | x(b) in A}.  We want symmetries to act on 2^L s.t. s(f(b)) = f(s(b)).
// let s = gy for g in D_4, y in L.  Given C in 2^L s.t. C = f(b), we have 
//   x in s(C) iff x in s(f(b)) = f(s(b))
//             iff x(s(b)) = (xgy)(b) = g(g'xg)y(b) in A iff (g'xgy)(b) in A
//             iff g'xgy in C
// Thus, we define
//   s(C) = gy(C) = {x in L | g'xgy in C} = {gzy'g' | z in C}
super_t transform_super(symmetry_t s, super_t super) __attribute__((const));

// Given b, find s minimizing s(b), and return s(b),s
tuple<board_t,symmetry_t> superstandardize(board_t board) __attribute__((const));
tuple<board_t,symmetry_t> superstandardize(side_t side0, side_t side1) __attribute__((const));

// A meaningless function invariant to global board transformations.  Extremely slow.
bool meaningless(board_t board, uint64_t salt=0) __attribute__((const));
super_t super_meaningless(board_t board, uint64_t salt=0) __attribute__((const));

#ifndef __wasm__
std::ostream& operator<<(std::ostream& output, symmetry_t s);
std::ostream& operator<<(std::ostream& output, local_symmetry_t s);
symmetry_t random_symmetry(Random& random);
#endif  // !__wasm__

// Convenient enumeration of all symmetries
struct symmetries_t {
  struct iter {
    int g;
    iter(int g) : g(g) {}
    void operator++() { g++; }
    bool operator!=(iter it) const { return g!=it.g; }
    symmetry_t operator*() const { return symmetry_t(g>>8,g&255); }
  };
  symmetries_t() {}
  int size() const { return 8*256; }
  iter begin() const { return iter(0); }
  iter end() const { return iter(size()); }
};
extern const symmetries_t symmetries;

}  // namespace pentago
namespace std {
template<> struct hash<pentago::symmetry_t> {
  auto operator()(pentago::symmetry_t s) const { return std::hash<int>()(s.global<<8 | s.local); }
};
}  // namespace std
