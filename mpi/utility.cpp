// MPI-related utilities

#include <pentago/mpi/utility.h>
namespace pentago {
namespace mpi {

using std::cout;
using std::cerr;
using std::endl;

static bool verbose_ = true;

bool verbose() {
  return verbose_;
}

void set_verbose(bool verbose) {
  verbose_ = verbose;
}

// Are we the master process
bool is_master() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  return !rank;
}

void die(const string& msg) {
  cerr << msg << endl;
  MPI_Abort(MPI_COMM_WORLD,1);
  abort();
}

void die_master(const string& msg) {
  if (!verbose())
    MPI_Barrier(MPI_COMM_WORLD);
  die(msg);
}

void check_failed(const char* file, const char* function, int line, const char* call, int result) {
  die(format("%s:%s:%d: %s failed with code %d",file,function,line,call,result));
}

scope_t::scope_t(const char* name)
  : thread_time_t(name), name(name) {
  if (verbose())
    cout << name << " : start" << endl;
}

scope_t::~scope_t() {
  if (verbose())
    cout << name << " : done" << endl;
}

}
}
