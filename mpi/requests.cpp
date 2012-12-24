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

void requests_t::add(MPI_Request request, const function<void(MPI_Status*)>& callback, bool cancellable) {
  requests.append(request);
  callbacks.push_back(callback);
  cancellables.append(cancellable);
}

void requests_t::checksome(bool wait) {
  vector<Tuple<function<void(MPI_Status*)>,MPI_Status*>> pending;
  int n = requests.size();
  MPI_Status statuses[n];
  {
    int indices[n];
    int finished;
    const auto check = wait?MPI_Waitsome:MPI_Testsome;
    {
      thread_time_t time(wait_kind,unevent);
      CHECK(check(n,requests.data(),&finished,indices,statuses));
    }
    // Add requested callback to pending list
    pending.resize(finished);
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
  }
  // Launch pending callbacks
  for (auto& pair : pending)
    pair.x(pair.y);
}

void requests_t::cancel_and_waitall() {
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
