// A list of requests together with callbacks

#include <pentago/mpi/requests.h>
#include <pentago/mpi/utility.h>
#include <pentago/utility/sort.h>
namespace pentago {
namespace mpi {

requests_t::requests_t() {}

requests_t::~requests_t() {
  if (requests.size())
    die("request_t destructing before all requests are complete.  Refusing to wait without being told.");
}

void requests_t::add(MPI_Request request, const function<void()>& callback) {
  requests.append(request);
  callbacks.push_back(callback);
}

bool requests_t::test() {
  int n = requests.size();
  int indices[n];
  int finished;
  CHECK(MPI_Testsome(n,requests.data(),&finished,indices,MPI_STATUSES_IGNORE));
  // The MPI standard doesn't appear to specify whether the indices are sorted.  Removing an unsorted list
  // of elements from an array is very unpleasant, so we sort.  If they're already sorted, this is cheap.
  insertion_sort(finished,indices);
  // Extract callbacks from the array without executing them
  for (int i=finished-1;i>=0;i--) {
    int j = indices[i];
    if (callbacks[j])
      callbacks[j](); // Execute callback
    requests.remove_index_lazy(j);
    callbacks[j].swap(callbacks[--n]);
  }
  callbacks.resize(n);
  return !n;
}

}
}
