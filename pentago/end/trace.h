// Tracing code for debugging purposes
#pragma once

#include "pentago/end/config.h"
#include "pentago/utility/format.h"
namespace pentago {
namespace end {

using std::string;

#if PENTAGO_MPI_TRACING
void mpi_trace(const string& msg);
#define PENTAGO_MPI_TRACE(...) mpi_trace(format(__VA_ARGS__))
#else
#define PENTAGO_MPI_TRACE(...) ((void)0)
#endif

// Called from the mpi_world_t constructor
void set_mpi_trace_rank(int rank);

}  // namespace end
}  // namespace pentago
