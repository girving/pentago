// Shuffled parallel_for using raw std::thread
//
// Iterates indices [0, n) in pseudorandom order across num_threads threads,
// using random_permute with a fixed key to scatter work across sections.
#pragma once

#include <cstddef>
#include <functional>
#include <semaphore>
#include <utility>
namespace pentago {

using std::function;
void parallel_for(const int num_threads, const size_t n, const function<void(size_t)>& f);

// Overlap I/O and compute: io_fn(i) runs under an I/O semaphore, then
// compute_fn(i, data) runs under a compute semaphore, with 2*num_threads
// threads so both can proceed concurrently.
template<class IO, class Compute>
static void overlapped_parallel_for(const int num_threads, const size_t n,
                                    const IO& io_fn, const Compute& compute_fn) {
  std::counting_semaphore<> io_sem(num_threads), compute_sem(num_threads);
  parallel_for(2 * num_threads, n, [&](const size_t i) {
    io_sem.acquire();
    auto data = io_fn(i);
    io_sem.release();
    compute_sem.acquire();
    compute_fn(i, std::move(data));
    compute_sem.release();
  });
}

}  // namespace pentago
