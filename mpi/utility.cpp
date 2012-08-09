// MPI related utilities

#include <pentago/mpi/utility.h>
#include <pentago/utility/ceil_div.h>
#include <pentago/utility/debug.h>
#include <other/core/utility/Log.h>
#include <other/core/utility/process.h>
namespace pentago {
namespace mpi {

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
  cerr << "\nrank " << comm_rank(MPI_COMM_WORLD) << ": " << msg << endl;
  process::backtrace();
  if (getenv("OTHER_BREAK_ON_ASSERT"))
    breakpoint();
  MPI_Abort(MPI_COMM_WORLD,1);
  abort();
}

void check_failed(const char* file, const char* function, int line, const char* call, int result) {
  die(format("%s:%s:%d: %s failed: %s",file,function,line,call,error_string(result)));
}

string error_string(int code) {
  int length;
  char error[MPI_MAX_ERROR_STRING];
  MPI_Error_string(code,error,&length);
  return error;
}

scope_t::scope_t(const char* name)
  : thread_time_t(name) {
  Log::push_scope(name);
}

scope_t::~scope_t() {
  Log::pop_scope();
}

Vector<int,4> section_blocks(section_t section) {
  return ceil_div(section.shape(),block_size);
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

MPI_Comm comm_dup(MPI_Comm comm) {
  MPI_Comm dup;
  CHECK(MPI_Comm_dup(comm,&dup));
  return dup;
}

int get_count(MPI_Status* status, MPI_Datatype datatype) {
  int count;
  CHECK(MPI_Get_count(status,datatype,&count));
  if (count == MPI_UNDEFINED) {
    CHECK(MPI_Get_count(status,MPI_BYTE,&count));
    die(format("get_count: MPI_Get_count result is undefined, bytes = %d",count));
  }
  return count;
}

mpi_world_t::mpi_world_t(int& argc, char**& argv) {
  int provided;
  CHECK(MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided));
  if (provided<MPI_THREAD_MULTIPLE)
    die(format("Insufficent MPI thread support: required = multiple, provided = %d",provided));

  // Call die instead of throwing exceptions from OTHER_ASSERT, OTHER_NOT_IMPLEMENTED, and THROW.
  debug::set_error_callback(die);
  throw_callback = die;

  // Make MPI errors return so that we can check the error codes ourselves
  MPI_Comm_set_errhandler(MPI_COMM_WORLD,MPI_ERRORS_RETURN);
}

mpi_world_t::~mpi_world_t() {
  CHECK(MPI_Finalize());
}

}
}
