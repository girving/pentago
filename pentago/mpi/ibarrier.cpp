// A nonblocking version of MPI_Barrier

#include <pentago/mpi/ibarrier.h>
#include <pentago/mpi/requests.h>
#include <pentago/mpi/utility.h>
#include <pentago/utility/debug.h>
#include <geode/math/integer_log.h>
namespace pentago {
namespace mpi {

static int parent(uint32_t rank) {
  return rank?rank^min_bit(rank):-1;
}

static void send_empty(int rank, int tag, MPI_Comm comm, requests_t& requests) {
  MPI_Request request;
  CHECK(MPI_Isend(0,0,MPI_INT,rank,tag,comm,&request));
  requests.free(request);
}

ibarrier_t::ibarrier_t(MPI_Comm comm, requests_t& requests, int tag)
  : comm(comm)
  , tag(tag)
  , requests(requests)
  , ranks(comm_size(comm))
  , rank(comm_rank(comm))
  , started_(false)
  , done_(false) {
  // How many children do we have?
  count = 1; // Count ourself as a child
  for (int jump = 1; rank+jump<ranks && !(rank&jump); jump *= 2)
    count++;
}

ibarrier_t::~ibarrier_t() {
  if (!done_)
    die("barrier destroyed before completion");
}

bool ibarrier_t::done() const {
  return done_;
}

void ibarrier_t::start() {
  GEODE_ASSERT(!started_ && !done_);
  started_ = true;
  decrement();
}

void ibarrier_t::recv() {
  // Receive the zero size message in order to know who it came from
  MPI_Status status;
  CHECK(MPI_Recv(0,0,MPI_INT,MPI_ANY_SOURCE,tag,comm,&status));
  process(status);
}

MPI_Request ibarrier_t::irecv() {
  MPI_Request request;
  CHECK(MPI_Irecv(0,0,MPI_INT,MPI_ANY_SOURCE,tag,comm,&request));
  return request;
}

void ibarrier_t::process(MPI_Status status) {
  const int source = status.MPI_SOURCE;
  if (source==parent(rank)) // Upwards message: everyone must be done
    set_done();
  else if (rank==parent(source)) // Downwards message: all our descendants are done
    decrement();
  else if (!rank && !source) // Special message from root to itself so that it suffices to check done() only after recv()
    /* pass */;
  else
    die("ibarrier_t: rank %d received unexpected message from rank %d (total ranks = %d)",rank,source,ranks);
}
    
void ibarrier_t::decrement() {
  GEODE_ASSERT(!done_ && (count>1 || started_));
  if (!--count) {
    if (!rank) { // We're the root, so all ranks have called start.  Begin upwards phase.
      set_done();
      // Send an extra message to ourselves so that it suffices to check done() only after recv()
      send_empty(0,tag,comm,requests);
    } else // Send started messages downwards
      send_empty(parent(rank),tag,comm,requests);
  }
}

void ibarrier_t::set_done() {
  GEODE_ASSERT(started_ && !count && !done_);
  done_ = true;
  // Send done messages upwards
  for (int jump = 1; rank+jump<ranks && !(rank&jump); jump *= 2)
    send_empty(rank+jump,tag,comm,requests);
}

/********************** ibarrier_countdown_t ***********************/

ibarrier_countdown_t::ibarrier_countdown_t(MPI_Comm comm, requests_t& requests, int tag, int count)
  : barrier(comm,requests,tag)
  , count(count) {
  GEODE_ASSERT(count>=0);
  if (!count)
    barrier.start();
}

ibarrier_countdown_t::~ibarrier_countdown_t() {}

void ibarrier_countdown_t::decrement() {
  GEODE_ASSERT(count>0);
  if (!--count)
    barrier.start();
}

}
}
