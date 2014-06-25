// MPI related utilities
#pragma once

#include <pentago/end/config.h>
#include <pentago/utility/thread.h>
#include <pentago/base/section.h>
#include <geode/utility/format.h>
#include <geode/vector/Vector.h>
#include <mpi.h>
namespace pentago {
namespace mpi {

using namespace geode;
using namespace pentago::end;
using std::string;

void GEODE_NORETURN(check_failed(const char* file, const char* function, int line, const char* call, int result));

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

// Init and finalize
struct mpi_world_t : public Noncopyable {
  mpi_world_t(int& argc, char**& argv);
  ~mpi_world_t();
};

// Map from C++ types to MPI types
template<class T> static inline MPI_Datatype datatype();
template<> inline MPI_Datatype datatype<double>()             { return MPI_DOUBLE; }
template<> inline MPI_Datatype datatype<int>()                { return MPI_INT; }
template<> inline MPI_Datatype datatype<long>()               { return MPI_LONG; }
template<> inline MPI_Datatype datatype<long long>()          { return MPI_LONG_LONG; }
template<> inline MPI_Datatype datatype<unsigned long>()      { return MPI_UNSIGNED_LONG; }
template<> inline MPI_Datatype datatype<unsigned long long>() { return MPI_UNSIGNED_LONG_LONG; }

}
}
