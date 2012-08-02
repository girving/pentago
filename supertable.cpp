// Rotation abstracted hashed transposition table

#include <pentago/supertable.h>
#include <pentago/stat.h>
#include <pentago/symmetry.h>
#include <pentago/trace.h>
#include <pentago/utility/debug.h>
#include <other/core/array/Array.h>
#include <other/core/python/module.h>
#include <other/core/random/Random.h>
#include <other/core/structure/Hashtable.h>
#include <other/core/utility/Log.h>
namespace pentago {

using namespace other;
using std::ostream;
using Log::cout;
using std::endl;

static const int hash_bits = 54;
static const int depth_bits = 10;
static_assert(hash_bits+depth_bits<=64,"");

struct superentry_t {
  uint64_t hash : hash_bits;
  uint64_t depth : depth_bits;
  superinfo_t info;
};

// Unfortunately, the alignment of __m128 is 16, so super_entry_t has 8 bytes
// of unused padding.  Maybe we can use it to store a lock in future?
static_assert(sizeof(superentry_t)==80,"");

// The current transposition table
static int table_bits = 0;
static Array<superentry_t> table;

void init_supertable(int bits) {
  if (bits<1 || bits>30)
    THROW(ValueError,"expected 1<=bits<=30, got bits = %d",bits);
  if (64-bits>hash_bits)
    THROW(ValueError,"bits = %d is too small, the high order hash bits won't fit",bits);
  table_bits = bits;
  cout << "initializing supertable: bits = "<<bits<<", size = "<<pow(2.,double(bits-20))*sizeof(superentry_t)<<"MB"<<endl;
  table = Array<superentry_t>((uint64_t)1<<bits,false);
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
  // Standardize the board
  superlookup_t data;
  board_t standard;
  superstandardize(side0,side1).get(standard,data.symmetry);
  data.hash = hash_board(standard|(uint64_t)aggressive<<aggressive_bit);
  // Lookup entry 
  const superentry_t& entry = table[data.hash&((1<<table_bits)-1)];
  if (entry.hash==data.hash>>table_bits) {
    superinfo_t& info = data.info;
    info = entry.info;
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

OTHER_NEVER_INLINE static void store_error(int depth, const superlookup_t& data, const superinfo_t& info) {
  const uint64_t board_flag = inverse_hash_board(data.hash);
  const board_t board = board_flag&~aggressive_mask;
  const bool aggressive = board_flag>>aggressive_bit;
  const superentry_t& entry = table[data.hash&((1<<table_bits)-1)];
  const superinfo_t& existing = entry.info;
  const super_t errors = (info.wins^existing.wins)&info.known&existing.known;
  const uint8_t r = first(errors);
  cout << format("inconsistency detected in super_store:\n  standard %lld, rotation %d, transformed %lld, aggressive %d\n  existing depth %d, existing value %d\n  new depth %d, new value %d",
    board,r,transform_board(symmetry_t(0,r),board),aggressive,entry.depth,existing.wins(r),depth,info.wins(r))<<endl;
  trace_error(aggressive,depth,board,"super_store");
}

template<bool aggressive> void super_store(int depth, const superlookup_t& data) {
  OTHER_ASSERT(data.hash);
  superentry_t& entry = table[data.hash&((1<<table_bits)-1)];
  if (entry.hash==data.hash>>table_bits || depth>=entry.depth) {
    superinfo_t info = data.info;
    superinfo_t& existing = entry.info;
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
  }
}

template void super_store<true>(int,const superlookup_t&);
template void super_store<false>(int,const superlookup_t&);

// This function makes inefficient use of random bits, so use for testing only
static int choice(Random& random, int choices=2) {
  return random.uniform<int>(0,choices);
}

static void supertable_test(int epochs) {
  // Prepare
  const int bits = 10;
  const int count = 1<<bits;
  init_supertable(bits);
  Ref<Random> random = new_<Random>(98312);
  uint64_t total_lookups = 0;

  // Repeatedly clear the supertable and run through a bunch of operations
  Hashtable<board_t> set;
  Array<board_t> boards;
  for (int epoch=0;epoch<epochs;epoch++) {
    // Start fresh
    clear_supertable();
    int successful_lookups = 0;

    // Generate a bunch of random boards
    Array<board_t> boards(count/8,false);
    for (board_t& board : boards)
      board = random_board(random);

    // Run a bunch of random table operations
    for (int step=0;step<count;step++) {
      // Pick a random board
      const int b = choice(random,boards.size());
      const symmetry_t symmetry = random_symmetry(random);
      const board_t board = transform_board(symmetry,boards[b]);
      const side_t side0 = unpack(board,0), side1 = unpack(board,1);
      const bool aggressive = b&1;
      superinfo_t info;
      info.known = random_super(random);
      info.wins = super_meaningless(board)&info.known;

      // If we're not at full depth, introduct some errors in favor of white 
      int depth = choice(random);
      if (!depth) {
        const super_t errors = super_meaningless(board,1831);
        info.wins = (aggressive?info.wins&~errors:info.wins|errors)&info.known;
      }

      // Verify that lookup returns consistent results
      total_lookups++;
      superlookup_t data = aggressive?super_lookup<true >(depth,side0,side1)
                                     :super_lookup<false>(depth,side0,side1);
      if (data.info.known) {
        OTHER_ASSERT(info.known&data.info.known); // should happen with overwhelming probability
        successful_lookups++;
      }
      super_t diff = (info.wins^data.info.wins)&info.known&data.info.known;
      if (!depth)
        diff &= aggressive?info.wins:~info.wins;
      OTHER_ASSERT(!diff);

      // Optionally store
      if (choice(random)) {
        data.info = info;
        aggressive?super_store<true >(depth,data)
                  :super_store<false>(depth,data);
      }
    }
    OTHER_ASSERT(successful_lookups>count/16);
  }
  cout << "total lookups = "<<total_lookups<<endl;
}

}
using namespace pentago;
using namespace other::python;

void wrap_supertable() {
  OTHER_FUNCTION(init_supertable)
  OTHER_FUNCTION(clear_supertable)
  OTHER_FUNCTION(supertable_bits)
  OTHER_FUNCTION(supertable_test)
}
