// A list of requests together with callbacks
#pragma once

#include <pentago/end/config.h>
#include <pentago/utility/spinlock.h>
#include <pentago/utility/job.h>
#include <other/core/array/Array.h>
#include <boost/noncopyable.hpp>
#include <boost/function.hpp>
#include <vector>
#include <mpi.h>
namespace pentago {
namespace mpi {

using namespace other;
using boost::function;
using std::vector;

/* Notes:
 *
 * 1. None of this is thread safe except for add_immediate.  Use the rest from the main communication thread.
 *
 * 2. Callbacks should be very cheap, since many of them can be called at once from test().
 *    If you want an expensive callback, use a cheap one that schedules an expensive function on a thread pool.
 *
 * 3. There is currently no routine to wait until all requests are complete, because in our current usage the
 *    global barrier automatically lets us know when everything is done.
 *
 * 4. It is safe to add requests during the callbacks.
 */

class requests_t : public boost::noncopyable {
  Array<MPI_Request> requests;
  vector<function<void(MPI_Status* status)>> callbacks;
  Array<bool> cancellables;
#if PENTAGO_MPI_FUNNEL
  // List of callbacks registered by other threads for immediate call.
  spinlock_t immediate_lock;
  vector<job_base_t*> immediates;
  volatile int immediate_count;
#endif
public:

  requests_t();
  ~requests_t();

  bool active() const {
    return requests.size();
  }

  // Register a new request with an optional callback, to be called when the request completes.
  // If the request can be safely cancelled, and the callback skipped, specify accordingly.
  void add(MPI_Request request, const function<void(MPI_Status* status)>& callback, bool cancellable=false);

  // A safer replacement for MPI_Request_free: register a cancellable request with no callback.
  // For why we don't use MPI_Request_free, see http://blogs.cisco.com/performance/mpi_request_free-is-evil.
  // Any buffers associated with the request can be safely used or freed once the requests_t destructs.
  void free(MPI_Request request);

  // Check for completed requests, blocking until at least one completes.
  // If !PENTAGO_MPI_FUNNEL, we do a single MPI_Waitsome; otherwise we poll between MPI_Testsome and immediate callbacks.
  void waitsome();

  // Cancel all pending messages, and wait for them to complete.  All pending callbacks
  // must be cancellable; otherwise an error is thrown.
  void cancel_and_waitall();

#if PENTAGO_MPI_FUNNEL
  // Register a callback to be called from the communication thread.  This is thread safe and safe to call from any thread.
  void add_immediate(job_t&& job) {
    spin_t spin(immediate_lock);
    immediates.push_back(job.release());
    immediate_count++;
  }
#endif
};

}
}
