// A nonblocking version of MPI_Barrier
//
// MPI contains the wonderfully useful routine MPI_Ibarrier, a nonblocking version
// of MPI_Barrier.  For why this is useful, see
//
//   http://scicomp.stackexchange.com/questions/2876/nonblocking-version-of-mpi-barrier-in-mpi-2
//
// Here, ibarrier_t is used to divide up the computation by slice, so that ranks know when
// they can safely deallocate the previous slice's data.
#pragma once

#include <pentago/end/config.h>
#include <boost/noncopyable.hpp>
#include <mpi.h>
namespace pentago {
namespace mpi {

class requests_t;
struct ibarrier_countdown_t;

// Warning: ibarrier_t is *not* thread safe
class ibarrier_t : public boost::noncopyable {
  const MPI_Comm comm;
  const int tag;
private:
  requests_t& requests;
  const int ranks;
  const int rank;
  bool started_;
  int count; // Number of incoming messages before we fire
  bool done_;
public:

  // Construct but don't yet activate the barrier
  ibarrier_t(MPI_Comm comm, requests_t& requests, int tag); 
  ~ibarrier_t();

  // Start the barrier operation.  Call only once.
  void start();

  // Are we finished?  It is sufficient to only check done after calls to recv (this avoids the need for some extra synchronization logic elsewhere).
  bool done() const;

  // If a probed message has the given tag, call this function.
  void recv();

  // Post a wildcard receive.  Once this completes, pass the result to status.
  MPI_Request irecv();

  // If you've already received a message with the given tag, call this function.
  void process(MPI_Status status);

private:
  void decrement();
  void set_done();
};

// When a count reaches zero, trigger an ibarrier.  ibarrier_countdown_t is *not* thread safe.
struct ibarrier_countdown_t : public boost::noncopyable {
  ibarrier_t barrier;
private:
  int count;
public:

  ibarrier_countdown_t(MPI_Comm comm, requests_t& requests, int tag, int count);
  ~ibarrier_countdown_t();

  int remaining() const {
    return count;
  }

  // Reduce the count
  void decrement();
};

}
}
