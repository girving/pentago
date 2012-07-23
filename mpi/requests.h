// A list of requests together with callbacks
#pragma once

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
 * 1. None of this is thread safe.  Use only from the main communication thread.
 *
 * 2. Callbacks should be very cheap, since many of them can be called at once from test().
 *    If you want an expensive callback, use a cheap one that schedules an expensive function on a thread pool.
 *
 * 3. There is currently no routine to wait until all requests are complete, because in our current usage the
 *    global barrier automatically lets us know when everything is done.
 */

class requests_t : public boost::noncopyable {
  Array<MPI_Request> requests;
  vector<function<void()>> callbacks;
public:

  requests_t();
  ~requests_t();

  // Register a new request with an optional callback, to be called when the request completes.
  void add(MPI_Request request, const function<void()>& callback);

  // Check for completed requests, and return true if everything is done.
  bool test();
};

}
}
