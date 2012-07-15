// MPI-related utilities

#include <pentago/thread.h>
#include <other/core/vector/Vector.h>
#include <mpi.h>
namespace pentago {
namespace mpi {

using std::string;

// Should we print?  Defaults to true.
bool verbose();
void set_verbose(bool verbose);

// Print a message and abort
void OTHER_NORETURN(die(const string& msg));

// If we're the master, print a message.  Then abort even if we're not
void OTHER_NORETURN(die_master(const string& msg));

void OTHER_NORETURN(check_failed(const char* file, const char* function, int line, const char* call, int result));

// Die if an MPI routine doesn't succeed
#define CHECK(call) ({ \
  int r_ = call; \
  if (r_ != MPI_SUCCESS) \
    check_failed(__FILE__,__FUNCTION__,__LINE,#call,r); \
  })

struct scope_t : public thread_time_t {
  const char* const name;

  scope_t(const char* name);
  ~scope_t();
};

template<int d> static Vector<int,d> block_shape(Vector<int,d> shape, Vector<int,d> block, int block_size) {
  return Vector<int,d>::componentwise_min(shape,block_size*(block+1))-block_size*block;
}

}
}
