// Store a bunch of arrays using compacting garbage collection to avoid fragmentation
#pragma once

/* I'm afraid of malloc fragmentation.  Storing compressed blocks involves
 * a large number of randomly sized arrays, the total size of which will
 * push up against the total memory in the machine.  Therefore, we implement
 * our own compacting collector to avoid any possibility of fragmentation.
 *
 * This class is thread safe for array accesses, but operations that change the group
 * structure must be serialized and separate from array operations.  However, while
 * multiple threads may safely compete to access a given array, this pattern is likely
 * not very useful.  Multiple simultaneous readers are supported once a group of arrays
 * is frozen, and in particular frozen arrays may be safely used in MPI_Isend calls.
 *
 * Locking discipline:
 *
 * We use the following ordering of locks:
 *
 *   heap_lock < array locks in order of group and index
 *
 * Threads are allocated to aquire any set of locks as long as they obey the above
 * ordering.  Here are the specific sets of locks held in different parts of the code:
 *
 *   1 array lock                : reading an array of modifying it in place.
 *   heap_lock                   : allocating new memory at the start of free space
 *   heap_lock + 1 array lock    : connecting an array to newly allocated memory
 *   heap_lock + all array locks : full garbage collection
 */

#include "pentago/end/config.h"
#include "pentago/utility/spinlock.h"
#include "pentago/utility/array.h"
namespace pentago {
namespace end {

using std::function;

class compacting_store_t : public boost::noncopyable {
public:
  class lock_t;

#ifdef __bgq__
  // See https://wiki.alcf.anl.gov/parts/index.php/Blue_Gene/Q#Allocating_Memory
  static constexpr int alignment = 128;
#else
  static constexpr int alignment = 64;
#endif

  static uint64_t align_size(const uint64_t size) {
    return (size+alignment-1)&~(alignment-1);
  }

  const uint64_t heap_size;
private:
  uint8_t* const heap_start;
  spinlock_t heap_lock; // See notes above
  uint64_t heap_next; // Next free index

  struct array_t {
    spinlock_t lock;
    int size;
    bool frozen; // If true, the array will remain immutable until the containing group destructs.
    uint8_t* data;
  };
  // The following is not locked: the user is responsible for serializing stucture changes.
  Vector<vector<array_t>,2> groups;

  // Callback called at the beginning and end of garbage collection for testing purposes
  const function<void()> collect_callback;

public:
  // Warning: The entire heap_size is allocated immediately upon construction.
  // If we run out, we die.  Choose wisely.
  compacting_store_t(const uint64_t heap_size, function<void()> collect_callback=nullptr);
  ~compacting_store_t();

  // These functions are essentially exact
  uint64_t memory_usage() const;
  static uint64_t memory_usage(const uint64_t arrays, const uint64_t heap_size);

  // For simplicity, arrays are divided into group, and arrays within each group
  // are numbered from 0 to n-1.  Under the hood, there are at most two groups.
  class group_t : public boost::noncopyable {
    friend class compacting_store_t::lock_t;
    const shared_ptr<compacting_store_t> store;
    int group;
  public:
    // Allocate a group with the given number of arrays.  All arrays start out empty.
    group_t(const shared_ptr<compacting_store_t>& store, const int count);
    ~group_t();

    // Freeze all arrays in this group.
    void freeze();

    // Grab the array in frozen form.  If the array is not frozen, we die.
    RawArray<const uint8_t> get_frozen(const int index) const;
  };

  // Lock access to a given array
  class lock_t : public boost::noncopyable {
    compacting_store_t& store;
    array_t* array;
  public:
    lock_t(group_t& group, const int index);
    ~lock_t();

    // Once we've locked the array, we can read or write it.
    RawArray<const uint8_t> get();
    void set(RawArray<const uint8_t> data);
  };

  // For testing purposes only
  const Vector<vector<array_t>,2>& groups_for_testing() const { return groups; }
  uint64_t heap_next_for_testing() const { return heap_next; }

private:
  friend struct array_before_t;

  // Perform a completely unlocked garbage collection.  The caller is solely responsible for safety.
  void unlocked_garbage_collect();

  // Assuming the heap_lock is already in hand, grab all array locks and perform a garbage collection.
  void partial_locked_garbage_collect();
};

}
}
