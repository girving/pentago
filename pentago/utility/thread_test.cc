#include "pentago/utility/curry.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/range.h"
#include "pentago/utility/spinlock.h"
#include "pentago/utility/thread.h"
#include "pentago/utility/threefry.h"
#include "pentago/utility/uint128.h"
#include "gtest/gtest.h"
namespace pentago {

static void add_noise(Array<uint128_t> data, int key, std::mutex* mutex, spinlock_t* spinlock) {
  const int n = data.size();
  for (const int i : range(n/2)) {
    std::lock_guard<std::mutex> lock(*mutex);
    data[i] += threefry(key, i);
  }
  for (const int i : range(n/2, n)) {
    spin_t spin(*spinlock);
    data[i] += threefry(key, i);
  }
  if (thread_type() == CPU)
    threads_schedule(IO, curry(add_noise, data, key+1, mutex, spinlock));
}

TEST(thread, pool) {
  init_threads(-1, -1);
  
  // Compute answers in serial
  const int jobs = 20;
  Array<uint128_t> serial(100);
  for (const int i : range(2*jobs))
    for (const int j : range(serial.size()))
      serial[j] += threefry(i,j);
  
  for (const int k __attribute__((unused)) : range(100)) {
    // Compute answers in parallel
    std::mutex mutex;
    spinlock_t spinlock;
    Array<uint128_t> parallel(serial.size());
    for (const int i : range(jobs))
      threads_schedule(CPU, curry(add_noise, parallel, 2*i, &mutex, &spinlock));
    threads_wait_all();

    // Compare
    ASSERT_EQ(serial, parallel);
  }
}

}
