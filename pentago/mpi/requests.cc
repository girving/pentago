// A list of requests together with callbacks

#include "pentago/mpi/requests.h"
#include "pentago/mpi/utility.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/sort.h"
namespace pentago {
namespace mpi {

using std::get;

requests_t::requests_t()
  : immediate_count(0) {}

requests_t::~requests_t() {
  if (requests.size())
    die("request_t destructing before %zu requests are complete.  Refusing to wait without being told",
        requests.size());
#if PENTAGO_MPI_FUNNEL
  if (immediate_count)
    die("request_t destructing before %d immediates are complete", int(immediate_count));
#endif
}

void requests_t::add(MPI_Request request, const function<void(MPI_Status*)>& callback,
                     bool cancellable) {
  requests.push_back(request);
  callbacks.push_back(callback);
  cancellables.push_back(cancellable);
}

void requests_t::free(MPI_Request request) {
  requests.push_back(request);
  callbacks.push_back(function<void(MPI_Status*)>());
  cancellables.push_back(true);
}

void requests_t::waitsome() {
  thread_time_t time(wait_kind,unevent);
  for (;;) {
    // Check for MPI request completions
    {
      int n = requests.size();
      GEODE_ASSERT(n);
      MPI_Status statuses[n];
      int indices[n];
      int finished;
      const auto check = PENTAGO_MPI_FUNNEL?MPI_Testsome:MPI_Waitsome;
      CHECK(check(n,requests.data(),&finished,indices,statuses));
      if (finished) { // If any requests finished, call their callbacks and return
        time.stop();
        // Add requested callback to pending list
        vector<tuple<function<void(MPI_Status*)>,MPI_Status*>> pending(finished);
        for (int i=0;i<finished;i++) {
          int j = indices[i];
          if (callbacks[j]) {
            swap(get<0>(pending[i]), callbacks[j]);
            get<1>(pending[i]) = &statuses[i];
          }
        }
        // The MPI standard doesn't appear to specify whether the indices are sorted.
        // Removing an unsorted list of elements from an array is very unpleasant, so we sort.
        // If they're already sorted, this is cheap.
        insertion_sort(finished,indices);
        // Prune away satisfied requests
        for (int i = finished-1; i >= 0; i--) {
          n--;
          const int j = indices[i];
          requests[j] = requests[n];
          callbacks[j].swap(callbacks[n]);
          cancellables[j] = cancellables[n];
        }
        requests.resize(n);
        callbacks.resize(n);
        cancellables.resize(n);
        // Launch pending callbacks
        for (auto& pair : pending)
          if (get<0>(pair))
            get<0>(pair)(get<1>(pair));
        return;
      } else if (!PENTAGO_MPI_FUNNEL)
        die("MPI_Waitsome completed with zero requests out of %d",n);
    }
#if PENTAGO_MPI_FUNNEL
    // Check for messages from worker threads
    if (immediate_count) {
      // Pull callbacks off shared pile
      immediate_lock.lock();
      const vector<function<void()>> ready(std::move(immediates));
      immediate_count = 0;
      immediate_lock.unlock();
      // If we have any, run them
      if (!ready.empty()) {
        time.stop();
        for (const auto& f : ready)
          f();
        return;
      }
    }
#endif
  }
}

void requests_t::cancel_and_waitall() {
#if PENTAGO_MPI_FUNNEL
  spin_t spin(immediate_lock);
  GEODE_ASSERT(!immediate_count);
#endif
  callbacks.clear();
  for (int i=0;i<requests.size();i++) {
    GEODE_ASSERT(cancellables[i]);
    CHECK(MPI_Cancel(&requests[i]));
  }
  CHECK(MPI_Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
  requests.clear();
  cancellables.clear();
}

}
}
