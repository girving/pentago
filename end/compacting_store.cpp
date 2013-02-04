// Store a bunch of arrays using compacting garbage collection to avoid fragmentation

#include <pentago/end/compacting_store.h>
#include <pentago/utility/debug.h>
#include <pentago/utility/large.h>
#include <pentago/utility/memory.h>
#include <pentago/utility/thread.h>
#include <other/core/array/sort.h>
#include <other/core/python/Class.h>
#include <other/core/random/Random.h>
#include <other/core/utility/curry.h>
#include <other/core/utility/Log.h>
#include <other/core/utility/str.h>
#include <sys/mman.h>
namespace pentago {
namespace end {

using Log::cout;
using std::endl;
OTHER_DEFINE_TYPE(compacting_store_t)

#ifdef __bgq__
// See https://wiki.alcf.anl.gov/parts/index.php/Blue_Gene/Q#Allocating_Memory
static const int alignment = 128;
#else
static const int alignment = 64;
#endif

static const uint64_t align_size(const uint64_t size) {
  return (size+alignment-1)&~(alignment-1);
}

compacting_store_t::compacting_store_t(const uint64_t heap_size_)
  : heap_size(align_size(heap_size_))
  , heap_start(heap_size?(uint8_t*)mmap(0,heap_size,PROT_READ|PROT_WRITE,MAP_ANON|MAP_PRIVATE,-1,0):0)
  , heap_next(0) {
  if (heap_start==MAP_FAILED)
    die("compacting_store_t: anonymous mmap of size %zu failed, %s",heap_size,strerror(errno));
  report_large_alloc(heap_size);
}

compacting_store_t::~compacting_store_t() {
  if (heap_start) {
    munmap(heap_start,heap_size);
    report_large_alloc(-heap_size);
  }
}

uint64_t compacting_store_t::memory_usage() const {
  return memory_usage(groups[0].size()+groups[1].size(),heap_size);
}

uint64_t compacting_store_t::memory_usage(const uint64_t arrays, const uint64_t heap_size) {
  return sizeof(array_t)*arrays+heap_size;
}

compacting_store_t::group_t::group_t(compacting_store_t& store, const int count)
  : store(ref(store)) {
  if (count<=0)
    group = -1;
  else {
    if (store.groups[0].size() && store.groups[1].size())
      die("compacting_store_t::group_t: no available groups");
    group = store.groups[0].size() ? 1 : 0;
    store.groups[group].resize(count,false,false);
    for (auto& array : store.groups[group]) {
      array.lock = spinlock_t();
      array.size = 0;
      array.frozen = false;
      array.data = 0;
    }
  }
}

compacting_store_t::group_t::~group_t() {
  if (group>=0)
    store->groups[group].clear();
}

void compacting_store_t::group_t::freeze() {
  Log::Scope scope("freeze");
  thread_time_t time(compacting_kind,unevent);
  if (group<0)
    return;
  if (store->groups[1-group].size())
    die("compacting_store_t::group_t::freeze: can't freeze a group while another group is allocated");
  store->unlocked_garbage_collect();
  for (auto& array : store->groups[group])
    array.frozen = true;
}

RawArray<const uint8_t> compacting_store_t::group_t::get_frozen(const int index) const {
  OTHER_ASSERT(group>=0);
  const auto arrays = store->groups[group].raw();
  OTHER_ASSERT(arrays.valid(index));
  const auto& array = arrays[index];
  if (!array.frozen)
    die("compacting_store_t::group_t::get_frozen: array %d is not frozen",index);
  return RawArray<const uint8_t>(array.size,array.data);
}

compacting_store_t::lock_t::lock_t(group_t& group, const int index)
  : store(group.store) {
  OTHER_ASSERT(group.group>=0);
  const auto arrays = store.groups[group.group].raw();
  OTHER_ASSERT(arrays.valid(index));
  array = &arrays[index];
  array->lock.lock();
}

compacting_store_t::lock_t::~lock_t() {
  array->lock.unlock();
}

RawArray<const uint8_t> compacting_store_t::lock_t::get() {
  return RawArray<const uint8_t>(array->size,array->data);
}

void compacting_store_t::lock_t::set(RawArray<const uint8_t> new_data) {
  const int asize = align_size(new_data.size());
  // Can we resize in place?
  if (asize > array->size) {
    // No: allocate a new block
#if !PENTAGO_MPI_COMPRESS
    // In uncompressed mode, allocations should always succeed, so we don't need to grab the heap_lock.
    const auto end = __sync_add_and_fetch(&store.heap_next,asize);
    if (end>store.heap_size)
      die("compacting_store_t::lock_t::set: ran out of memory in uncompressed mode: old size = %d, new size = %d",array->size,new_data.size());
    array->data = store.heap_start+end-asize;
#else
    // First, deallocate the array and release the lock to preserve locking discipline.
    array->size = 0;
    array->data = 0;
    array->lock.unlock();
    // Grab the master heap lock, and either allocate or garbage collect
    spin_t heap_spin(store.heap_lock);
    if (store.heap_next+asize>store.heap_size) {
      // Insufficent space: perform a garbage collection
      store.partial_locked_garbage_collect();
      // Require enough space after the collection
      if (store.heap_next+asize>store.heap_size)
        die("compacting_store_t::lock_t::set: insufficient space even after garbage collection: heap size = %s, free = %d, new size = %d",large(store.heap_size),store.heap_size-store.heap_next,new_data.size());
    }
    array->lock.lock();
    array->data = store.heap_start+store.heap_next;
    store.heap_next += asize;
#endif
  }
  // Finally, copy the data into place
  memcpy(array->data,new_data.data(),new_data.size());
  array->size = new_data.size();
}

struct array_before_t {
  bool operator()(const compacting_store_t::array_t* a, const compacting_store_t::array_t* b) {
    return a->data < b->data;
  }
};

void compacting_store_t::unlocked_garbage_collect() {
  if (collect_callback)
    collect_callback();
  // Collect pointers to all arrays
  Array<array_t*> arrays;
  arrays.preallocate(groups[0].size()+groups[1].size());
  for (auto& group : groups)
    for (auto& array : group)
      arrays.append_assuming_enough_space(&array);
  // Sort in order of their position within the heap
  sort(arrays,array_before_t());
  // Compact
  uint8_t* target = heap_start;
  for (array_t* array : arrays) {
    if (target != array->data) {
      if (array->frozen)
        die("compacting_store_t::garbage_collect: can't move a frozen array");
      memmove(target,array->data,array->size);
      array->data = target;
    }
    target += align_size(array->size);
  }
  // Initialize the next allocation cycle
  heap_next = target-heap_start;
  OTHER_ASSERT(heap_next<=heap_size);
  const uint64_t free = heap_size-heap_next;
  const double ratio = double(free)/heap_size;
  cout << format("collection: free ratio = %g",ratio) << endl;
  if (ratio<compacting_store_min_free_ratio)
    die("compacting_store_t::garbage_collect: insufficient free space after garbage collection: heap size = %s, free = %s, free ratio = %g (required = %g)",large(heap_size),large(free),ratio,compacting_store_min_free_ratio);
  if (collect_callback)
    collect_callback();
}

// The caller is responsible for holding the heap_lock around this call
void compacting_store_t::partial_locked_garbage_collect() {
  // Grab all array locks
  for (auto& group : groups)
    for (auto& array : group)
      array.lock.lock();
  // Run the collection
  unlocked_garbage_collect();
  // Release all locks (this can safely be done in any order)
  for (auto& group : groups)
    for (auto& array : group)
      array.lock.unlock();
}

struct thrasher_t {
  static const int jobs = 32;
  static const int chunks = jobs;
  static const int arrays = 16;
  static const int iterations = 1024;
  static const bool verbose = false;

  const Ref<compacting_store_t> store;
  compacting_store_t::group_t group;
  const uint64_t limit;
  spinlock_t used_lock;
  uint64_t used;

  thrasher_t()
    : store(new_<compacting_store_t>(chunks*alignment+1))
    , group(store,arrays)
    , limit(.9*store->heap_size)
    , used(0) {
    cout << "heap size = "<<store->heap_size<<", limit = "<<limit<<endl;
    store->collect_callback = curry(&thrasher_t::unlocked_check,this);
    for (const int key : range(jobs))
      threads_schedule(CPU,curry(&thrasher_t::thrash,this,key));
    threads_wait_all();
  }

  static uint8_t sig(RawArray<const uint8_t> data) {
    uint8_t s = 0;
    for (auto& c : data)
      s ^= c;
    return s;
  }

  static string hex(RawArray<const uint8_t> data) {
    string s;
    for (const uint8_t c : data)
      s += format("%x%x",c&15,c>>4);
    return s;
  }

  void unlocked_check() {
    if (verbose)
      cout << "check: used = "<<used<<endl;
    for (auto& group : store->groups)
      for (const int array : range(group.size()))
        if (group[array].size) {
          RawArray<const uint8_t> data(group[array].size,group[array].data);
          if (verbose)
            cout << format("  in check %d = '%s'",array,hex(data)) << endl;
          OTHER_ASSERT(sig(data)==7);
        }
    used = min(used,store->heap_next);
  }

  void thrash(const int key) {
    const auto random = new_<Random>(key);
    for (int iter=0;iter<iterations;iter++) {
      // Lock a random array
      const int array = random->uniform<int>(0,arrays);
      {
        // Verify that the contents xor to 7
        compacting_store_t::lock_t alock(group,array);
        if (alock.get().size()) {
          if (verbose)
            cout << format("get %d = '%s'",array,hex(alock.get())) << endl;
          OTHER_ASSERT(sig(alock.get())==7);
        }
      }
      // Generate a new random array xoring to 7
      const Array<uint8_t> data(random->uniform<int>(1,2*alignment),false);
      {
        uint8_t s = 0;
        for (const int i : range(1,data.size()))
          s ^= (data[i] = random->bits<uint8_t>());
        data[0] = 7^s;
      }
      // If there's space, set the new array, otherwise clear the existing one
      compacting_store_t::lock_t alock(group,array);
      const int old_asize = align_size(alock.get().size());
      const int diff = align_size(data.size())-old_asize;
      used_lock.lock();
      if (used+diff<=limit) {
        used += diff;
        used_lock.unlock();
        if (verbose)
          cout << format("set %d = '%s'",array,hex(data)) << endl;
        alock.set(data);
      } else {
        used -= old_asize;
        used_lock.unlock();
        if (verbose)
          cout << format("clear %d",array) << endl;
        alock.set(Array<uint8_t>());
      }
    }
  }
};

static void compacting_store_test() {
  thrasher_t();
}

}
}
using namespace pentago::end;

void wrap_compacting_store() {
  typedef compacting_store_t Self;
  Class<Self>("compacting_store_t")
    .OTHER_INIT(uint64_t)
    ;
  OTHER_FUNCTION(compacting_store_test)
}
