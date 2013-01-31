// MPI related utilities
#pragma once

#include <pentago/end/config.h>
#include <pentago/utility/thread.h>
#include <pentago/base/section.h>
#include <other/core/utility/format.h>
#include <other/core/vector/Vector.h>
#include <mpi.h>
namespace pentago {
namespace mpi {

using namespace other;
using namespace pentago::end;
using std::string;

void OTHER_NORETURN(check_failed(const char* file, const char* function, int line, const char* call, int result));

// Die if an MPI routine doesn't succeed
#define CHECK(call) ({ \
  int r_ = call; \
  if (r_ != MPI_SUCCESS) \
    check_failed(__FILE__,__FUNCTION__,__LINE__,#call,r_); \
  })

// Convert an error code into a string
string error_string(int code);

// Convenience functions
int comm_size(MPI_Comm comm);
int comm_rank(MPI_Comm comm);
MPI_Comm comm_dup(MPI_Comm comm);
int get_count(MPI_Status* status, MPI_Datatype datatype);
void send_empty(int rank, int tag, MPI_Comm comm); // Send an empty message

// Init and finalize
struct mpi_world_t : public boost::noncopyable {
  mpi_world_t(int& argc, char**& argv);
  ~mpi_world_t();
};

}
}
