// Rotation abstracted hashed transposition table
//
// Symmetry-aware transposition tables.  Entries exist only in superstandardized
// form, and store two super_t's: a mask of which entries we know and the values
// of those we do (recall that one player is considered "aggressive" and loses ties,
// so only one bit is needed per position).  Some logic is required when combining
// values of different depths together: information of low depth can be used if it
// signified an immediate end to the game, otherwise not.
#pragma once

#include <pentago/base/hash.h>
#include <pentago/base/superscore.h>
#include <pentago/base/symmetry.h>
namespace pentago {

// We use the high bit to mark whether the player to move is the aggressor (the player trying to win)
static const int aggressive_bit = 63;
static const uint64_t aggressive_mask = (uint64_t)1<<aggressive_bit;

// Initialize a empty table with 1<<bits entries
extern Unit init_supertable(int bits);

// Clear all supertable entries
Unit clear_supertable();

// lg(entries) or 0 for uninitialized
extern int supertable_bits() GEODE_PURE;

// Structure to feed information from a lookup to its corresponding store
struct superlookup_t {
  uint64_t hash;
  symmetry_t symmetry;
  superinfo_t info;
};

// Look up an entry in the table to at least the given depth
template<bool aggressive> extern superlookup_t super_lookup(int depth, side_t side0, side_t side1) GEODE_PURE;

// Store new data in the table.  The data structure should be the same structure returned by
// super_lookup, with possibly more known information in info.
template<bool aggressive> extern void super_store(int depth, const superlookup_t& data);

}
