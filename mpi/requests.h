// A list of requests together with callbacks
#pragma once

#include <pentago/mpi/config.h>
#include <other/core/array/Array.h>
#include <boost/noncopyable.hpp>
#include <boost/function.hpp>
#include <vector>
namespace pentago {
namespace mpi {

using namespace other;
using boost::function;
using std::vector;

/* Notes:
 *
 * 1. None of this is thread safe.  Use only from the main communication thread.
 *
 * 2. Callbacks should be very cheap, since many of them can be called at once from test().
 *    If you want an expensive callback, use a cheap one that schedules an expensive function on a thread pool.
 *
 * 3. There is currently no routine to wait until all requests are complete, because in our current usage the
 *    global barrier automatically lets us know when everything is done.
 *
 * 4. It is safe to add requests during the callbacks
 */

class requests_t : public boost::noncopyable {
  Array<MPI_Request> requests;
  vector<function<void(MPI_Status* status)>> callbacks;
  Array<bool> cancellables;
public:

  requests_t();
  ~requests_t();

  bool active() const {
    return requests.size();
  }

  // Register a new request with an optional callback, to be called when the request completes.
  // If the request can be safely cancelled, and the callback skipped, specify accordingly.
  void add(MPI_Request request, const function<void(MPI_Status* status)>& callback, bool cancellable=false);

  // Check for completed requests without blocking
  void testsome() {
    checksome(false);
  }

  // Check for completed requests, blocking until at least one completes
  void waitsome() {
    checksome(true);
  }

  // Cancel all pending messages, and wait for them to complete.  All pending callbacks
  // must be cancellable; otherwise an error is thrown.
  void cancel_and_waitall();

private:
  void checksome(bool wait); 
};

}
}
