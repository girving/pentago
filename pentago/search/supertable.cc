// Rotation abstracted hashed transposition table

#include "pentago/search/supertable.h"
#include "pentago/search/stat.h"
#include "pentago/base/symmetry.h"
#include "pentago/search/trace.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/array.h"
#include "pentago/utility/random.h"
#include "pentago/utility/log.h"
namespace pentago {

using std::max;
using std::tie;

static const int hash_bits = 54;
static const int depth_bits = 10;
static_assert(hash_bits+depth_bits<=64,"");

// Unfortunately, the alignment of __m128 is 16, so superentry_t would 8 bytes
// of unused padding if it included a superinfo_t.  memcpy to the rescue!
struct compact_superinfo_t {
  uint64_t x[sizeof(superinfo_t)/sizeof(uint64_t)];
};

struct superentry_t {
  uint64_t hash : hash_bits;
  uint64_t depth : depth_bits;
  compact_superinfo_t info;
};
static_assert(sizeof(superentry_t)==72,"");

template<class D,class S> static inline D mcast(const S& src) {
  static_assert(sizeof(S)==sizeof(D),"");
  D dst;
  memcpy(&dst,&src,sizeof(S));
  return dst;
}

// The current transposition table
static int table_bits = 0;
static Array<superentry_t> table;

void init_supertable(int bits) {
  if (bits<1 || bits>30)
    THROW(ValueError,"expected 1<=bits<=30, got bits = %d",bits);
  if (64-bits>hash_bits)
    THROW(ValueError,"bits = %d is too small, the high order hash bits won't fit",bits);
  table_bits = bits;
  slog("initializing supertable: bits = %d, size = %gMB", 
       bits, pow(2.,double(bits-20))*sizeof(superentry_t));
  table = Array<superentry_t>(1<<bits,uninit);
  clear_supertable();
}

void clear_supertable() {
  memset(table.data(),0,sizeof(superentry_t)*table.size());
  TRACE(trace_restart());
}

int supertable_bits() {
  return table_bits;
}

template<bool aggressive> superlookup_t super_lookup(int depth, side_t side0, side_t side1) {
  STAT(total_lookups++);
  STAT_DETAIL(lookup_detail[depth]++);
  // Standardize the board
  superlookup_t data;
  board_t standard;
  tie(standard, data.symmetry) = superstandardize(side0, side1);
  data.hash = hash_board(standard|(uint64_t)aggressive<<aggressive_bit);
  // Lookup entry 
  const superentry_t& entry = table[data.hash&((1<<table_bits)-1)];
  if (entry.hash==data.hash>>table_bits) {
    superinfo_t& info = data.info;
    info = mcast<superinfo_t>(entry.info);
    // Prepare to transform: wins(b) = wins(s'(s(b))) = s'(wins(s(b)))
    const symmetry_t si = data.symmetry.inverse();
    // If we don't have enough depth, we can only use wins for black or losses for white
    if (depth>entry.depth) {
      info.known &= aggressive?info.wins:~info.wins;
      TRACE(trace_dependency(depth,pack(side0,side1),entry.depth,pack(side0,side1),superinfo_t(info.known,aggressive?info.known:~info.known)));
      info.known = transform_super(si,info.known); // In this case we get away with only one transform call
      info.wins = aggressive?info.known:super_t(0);
    } else {
      STAT(successful_lookups++);
      STAT_DETAIL(successful_lookup_detail[depth]++);
      info.known = transform_super(si,info.known);
      info.wins  = transform_super(si,info.wins);
    }
  } else {
    // Nothing found
    data.info = superinfo_t(0);
  }
  return data;
}

template superlookup_t super_lookup<true>(int,side_t,side_t);
template superlookup_t super_lookup<false>(int,side_t,side_t);

__attribute__((noinline)) static void store_error(int depth, const superlookup_t& data,
                                                  const superinfo_t& info) {
  const uint64_t board_flag = inverse_hash_board(data.hash);
  const board_t board = board_flag&~aggressive_mask;
  const bool aggressive = board_flag>>aggressive_bit;
  const superentry_t& entry = table[data.hash&((1<<table_bits)-1)];
  const auto existing = mcast<superinfo_t>(entry.info);
  const super_t errors = (info.wins^existing.wins)&info.known&existing.known;
  const uint8_t r = first(errors);
  slog("inconsistency detected in super_store:\n  standard %lld, rotation %d, transformed %lld, "
       "aggressive %d\n  existing depth %d, existing value %d\n  new depth %d, new value %d",
       board, r, transform_board(symmetry_t(0,r),board), aggressive, entry.depth, existing.wins(r),
       depth, info.wins(r));
  trace_error(aggressive,depth,board,"super_store");
}

template<bool aggressive> void super_store(int depth, const superlookup_t& data) {
  GEODE_ASSERT(data.hash);
  superentry_t& entry = table[data.hash&((1<<table_bits)-1)];
  if (entry.hash==data.hash>>table_bits || depth>=entry.depth) {
    superinfo_t info = data.info;
    superinfo_t existing = mcast<superinfo_t>(entry.info);
    // Transform: wins(s(b)) = s(wins(b))
    info.known = transform_super(data.symmetry,info.known);
    info.wins  = transform_super(data.symmetry,info.wins);
    // Insert new entry or merge with the existing one
    if (entry.hash==data.hash>>table_bits) {
      // Discard low depth information
      int max_depth = max(depth,(int)entry.depth); 
      if (depth<max_depth) {
        super_t mask = aggressive?info.wins:~info.wins;
        info.known &= mask;
        info.wins &= mask;
        TRACE(trace_dependency(max_depth,inverse_hash_board(data.hash),depth,inverse_hash_board(data.hash),info));
        TRACE(trace_check(max_depth,inverse_hash_board(data.hash),info,"super_store.new"));
      }
      if (entry.depth<max_depth) {
        super_t mask = aggressive?existing.wins:~existing.wins;
        existing.known &= mask;
        existing.wins &= mask;
        TRACE(trace_dependency(max_depth,inverse_hash_board(data.hash),entry.depth,inverse_hash_board(data.hash),existing));
        TRACE(trace_check(max_depth,inverse_hash_board(data.hash),existing,"super_store.existing"));
      }
      // Common knowledge should match.  Unfortunately, this assertion can generate false positives in cases where the table is
      // "polluted" with information with higher than reported depth.  However, I'm going to leave it in, since the false positives
      // never occur if the table depth always increases, and it's important to have validity checks wherever possible.
      if ((info.wins^existing.wins)&info.known&existing.known)
        store_error(depth,data,info);
      // Merge information
      entry.depth = max_depth;
      existing.known |= info.known;
      existing.wins ^= (existing.wins^info.wins)&info.known;
    } else /* depth>=entry.depth */ {
      // Write a new entry
      entry.hash = data.hash>>table_bits;
      entry.depth = depth;
      existing = info;
    }
    entry.info = mcast<compact_superinfo_t>(existing);
  }
}

template void super_store<true>(int,const superlookup_t&);
template void super_store<false>(int,const superlookup_t&);

}
