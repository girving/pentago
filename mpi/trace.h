// Tracing code for debugging purposes
#pragma once

#include <other/core/utility/format.h>
namespace pentago {
namespace mpi {

using std::string;

// Uncomment to enable expensive consistency checking
//#define MPI_DEBUG

// Uncomment to enable MPI tracing
//#define MPI_TRACING

#ifdef MPI_TRACING
void mpi_trace(const string& msg);
#define MPI_TRACE(...) mpi_trace(format(__VA_ARGS__))
#else
#define MPI_TRACE(...) ((void)0)
#endif

}
}
