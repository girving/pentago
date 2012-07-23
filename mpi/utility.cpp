// MPI related utilities

#include <pentago/mpi/utility.h>
#include <other/core/utility/Log.h>
namespace pentago {
namespace mpi {

using Log::cout;
using Log::cerr;
using std::endl;

static bool verbose_ = true;

bool verbose() {
  return verbose_;
}

void set_verbose(bool verbose) {
  verbose_ = verbose;
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
  : thread_time_t(name) {
  Log::push_scope(name);
}

scope_t::~scope_t() {
  Log::pop_scope();
}

Vector<int,4> section_blocks(section_t section, int block_size) {
  return (section.shape()+block_size-1)/block_size;
}

int comm_size(MPI_Comm comm) {
  int size;
  CHECK(MPI_Comm_size(comm,&size));
  return size;
}

int comm_rank(MPI_Comm comm) {
  int rank;
  CHECK(MPI_Comm_rank(comm,&rank));
  return rank;
}

}
}
