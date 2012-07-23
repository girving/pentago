// A nonblocking version of MPI_Barrier

#include <mpi.h>
#include <boost/noncopyable.hpp>
namespace pentago {
namespace mpi {

struct ibarrier_countdown_t;

// Warning: ibarrier_t is *not* thread safe
class ibarrier_t : public boost::noncopyable {
  const MPI_Comm comm;
  const int tag;
private:
  const int ranks;
  const int rank;
  bool started_;
  int count; // Number of incoming messages before we fire
  bool done_;
public:

  // Construct but don't yet activate the barrier
  ibarrier_t(MPI_Comm comm, int tag); 
  ~ibarrier_t();

  // Start the barrier operation.  Call only once.
  void start();

  // Are we finished?  It is sufficient to only check done after calls to recv (this avoids the need for some extra synchronization logic elsewhere).
  bool done() const;

  // If a probed message has the given tag, call this function.
  void recv();

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

  ibarrier_countdown_t(MPI_Comm comm, int tag, int count);
  ~ibarrier_countdown_t();

  // Reduce the count
  void decrement();
};

}
}
