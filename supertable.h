// Rotation abstracted hashed transposition table

#pragma once

#include "table.h"
#include "superscore.h"
#include "symmetry.h"
namespace pentago {

// Initialize a empty table with 1<<bits entries
extern void init_supertable(int bits);

// Clear all supertable entries
void clear_supertable();

// lg(entries) or 0 for uninitialized
extern int supertable_bits() OTHER_PURE;

// Structure to feed information from a lookup to its corresponding store
struct superlookup_t {
  uint64_t hash;
  symmetry_t symmetry;
  superinfo_t info;
};

// Look up an entry in the table to at least the given depth
template<bool black> extern superlookup_t super_lookup(int depth, side_t side0, side_t side1) OTHER_PURE;

// Store new data in the table.  The data structure should be the same structure returned by
// super_lookup, with possibly more known information in info.
template<bool black> extern void super_store(int depth, const superlookup_t& data);

}
