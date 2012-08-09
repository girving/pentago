// MPI related utilities

#include <pentago/mpi/config.h>
#include <pentago/thread.h>
#include <pentago/section.h>
#include <other/core/vector/Vector.h>
namespace pentago {
namespace mpi {

using std::string;

// Should we print?  Defaults to true.
bool verbose();
void set_verbose(bool verbose);

// Print a message and abort
void OTHER_NORETURN(die(const string& msg));

void OTHER_NORETURN(check_failed(const char* file, const char* function, int line, const char* call, int result));

// Die if an MPI routine doesn't succeed
#define CHECK(call) ({ \
  int r_ = call; \
  if (r_ != MPI_SUCCESS) \
    check_failed(__FILE__,__FUNCTION__,__LINE__,#call,r_); \
  })

// Convert an error code into a string
string error_string(int code);

struct scope_t : public thread_time_t {
  scope_t(const char* name);
  ~scope_t();
};

Vector<int,4> section_blocks(section_t section);

template<int d> static inline Vector<int,d> block_shape(Vector<int,d> shape, Vector<int,d> block) {
  return Vector<int,d>::componentwise_min(shape,block_size*(block+1))-block_size*block;
}

// Convenience functions
int comm_size(MPI_Comm comm);
int comm_rank(MPI_Comm comm);
MPI_Comm comm_dup(MPI_Comm comm);
int get_count(MPI_Status* status, MPI_Datatype datatype);

// Init and finalize
struct mpi_world_t : public boost::noncopyable {
  mpi_world_t(int& argc, char**& argv);
  ~mpi_world_t();
};

}
}
