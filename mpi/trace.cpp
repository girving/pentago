// Tracing code for debugging purposes

#include <pentago/mpi/trace.h>
#include <pentago/mpi/utility.h>
#include <pentago/utility/spinlock.h>
#include <other/core/utility/Log.h>
namespace pentago {
namespace mpi {

using std::endl;

#if PENTAGO_MPI_TRACING

static spinlock_t lock;

void mpi_trace(const string& msg) {
  const int rank = comm_rank(MPI_COMM_WORLD);
  spin_t spin(lock);
  Log::cout << "trace "<<rank<<": "<<msg<<endl;
}

#endif

}
}
