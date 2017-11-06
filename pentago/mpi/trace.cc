// Tracing code for debugging purposes

#include "pentago/mpi/trace.h"
#include "pentago/utility/spinlock.h"
namespace pentago {
namespace mpi {

static int rank = -1;

void set_mpi_trace_rank(int r) {
  rank = r;
}

#if PENTAGO_MPI_TRACING

void mpi_trace(const string& msg) {
  slog("trace %d: %s", rank, msg);
}

#endif

}
}
