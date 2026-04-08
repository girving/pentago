// Shuffled parallel_for using raw std::thread
//
// Iterates indices [0, n) in pseudorandom order across num_threads threads,
// using random_permute with a fixed key to scatter work across sections.
#pragma once

#include <cstddef>
#include <functional>
namespace pentago {

using std::function;

void parallel_for(const int num_threads, const size_t n, const function<void(size_t)>& f);

}  // namespace pentago
