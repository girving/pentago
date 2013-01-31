// A list of requests together with callbacks

#include <pentago/mpi/requests.h>
#include <pentago/mpi/utility.h>
#include <pentago/utility/debug.h>
#include <pentago/utility/sort.h>
namespace pentago {
namespace mpi {

requests_t::requests_t() {}

requests_t::~requests_t() {
  if (requests.size())
    die("request_t destructing before %zu requests are complete.  Refusing to wait without being told",requests.size());
#if PENTAGO_MPI_FUNNEL
  if (immediates.size())
    die("request_t destructing before %zu immediates are complete",immediates.size());
#endif
}

void requests_t::add(MPI_Request request, const function<void(MPI_Status*)>& callback, bool cancellable) {
  requests.append(request);
  callbacks.push_back(callback);
  cancellables.append(cancellable);
}

void requests_t::waitsome() {
  thread_time_t time(wait_kind,unevent);
  for (;;) {
    // Check for MPI request completions
    {
      int n = requests.size();
      OTHER_ASSERT(n);
      MPI_Status statuses[n];
      int indices[n];
      int finished;
      const auto check = PENTAGO_MPI_FUNNEL?MPI_Testsome:MPI_Waitsome;
      CHECK(check(n,requests.data(),&finished,indices,statuses));
      if (finished) { // If any requests finished, call their callbacks and return
        time.stop();
        // Add requested callback to pending list
        vector<Tuple<function<void(MPI_Status*)>,MPI_Status*>> pending(finished);
        for (int i=0;i<finished;i++) {
          int j = indices[i];
          if (callbacks[j]) {
            swap(pending[i].x,callbacks[j]);
            pending[i].y = &statuses[i];
          }
        }
        // The MPI standard doesn't appear to specify whether the indices are sorted.  Removing an unsorted list
        // of elements from an array is very unpleasant, so we sort.  If they're already sorted, this is cheap.
        insertion_sort(finished,indices);
        // Prune away satisfied requests
        for (int i=finished-1;i>=0;i--) {
          int j = indices[i];
          requests.remove_index_lazy(j);
          cancellables.remove_index_lazy(j);
          callbacks[j].swap(callbacks[--n]);
        }
        callbacks.resize(n);
        // Launch pending callbacks
        for (auto& pair : pending)
          pair.x(pair.y);
        return;
      } else if (!PENTAGO_MPI_FUNNEL)
        die("MPI_Waitsome completed with zero requests out of %d",n);
    }
#if PENTAGO_MPI_FUNNEL
    // Check for messages from worker threads
    { 
      // Pull callbacks off shared pile
      immediate_lock.lock();
      const int n = immediates.size();
      job_base_t* ready[n];
      if (n)
        memcpy(ready,&immediates[0],sizeof(job_base_t*)*n);
      immediates.clear();
      immediate_lock.unlock();
      // If we have any, run them
      if (n) {
        time.stop();
        // If exceptions are thrown here there'd be a memory leak, but we don't allow exceptions.
        for (const auto job : ready) {
          (*job)();
          delete job;
        }
        return;
      }
    }
#endif
  }
}

void requests_t::cancel_and_waitall() {
#if PENTAGO_MPI_FUNNEL
  spin_t spin(immediate_lock);
  OTHER_ASSERT(!immediates.size());
#endif
  callbacks.clear();
  for (int i=0;i<requests.size();i++) {
    OTHER_ASSERT(cancellables[i]);
    CHECK(MPI_Cancel(&requests[i]));
  }
  CHECK(MPI_Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
  requests.clear();
  cancellables.clear();
}

}
}
