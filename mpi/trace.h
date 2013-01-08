// Tracing code for debugging purposes
#pragma once

#include <pentago/mpi/config.h>
#include <other/core/utility/format.h>
namespace pentago {
namespace mpi {

using std::string;

#if PENTAGO_MPI_TRACING
void mpi_trace(const string& msg);
#define PENTAGO_MPI_TRACE(...) mpi_trace(format(__VA_ARGS__))
#else
#define PENTAGO_MPI_TRACE(...) ((void)0)
#endif

}
}
