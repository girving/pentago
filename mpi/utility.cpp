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

void check_failed(const char* file, const char* function, int line, const char* call, int result) {
  die("%s:%s:%d: %s failed: %s",file,function,line,call,error_string(result));
}

string error_string(int code) {
  int length;
  char error[MPI_MAX_ERROR_STRING];
  MPI_Error_string(code,error,&length);
  return error;
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
    die("get_count: MPI_Get_count result is undefined, bytes = %d",count);
  }
  return count;
}

static string str_thread_support(const int support) {
  switch (support) {
    case MPI_THREAD_SINGLE: return "single";
    case MPI_THREAD_FUNNELED: return "funneled";
    case MPI_THREAD_SERIALIZED: return "serialized";
    case MPI_THREAD_MULTIPLE: return "multiple";
    default: return format("unknown (%d)",support);
  }
}

static void OTHER_NORETURN(mpi_die_helper(const string& msg)) OTHER_COLD;
static void                mpi_die_helper(const string& msg) {
  cerr << "\nrank " << comm_rank(MPI_COMM_WORLD) << ": " << msg << endl;
  process::backtrace();
  if (getenv("OTHER_BREAK_ON_ASSERT"))
    breakpoint();
  // Ideally we would call MPI_Abort here, but that's technically disallowed if we're not
  // running in MPI_THREAD_MULTIPLE mode.  Therefore, we bail more forcefully.
  abort();
}

mpi_world_t::mpi_world_t(int& argc, char**& argv) {
  const int required = PENTAGO_MPI_FUNNEL?MPI_THREAD_FUNNELED:MPI_THREAD_MULTIPLE;
  int provided;
  CHECK(MPI_Init_thread(&argc,&argv,required,&provided));
  if (provided<required)
    die("Insufficent MPI thread support: required = %s, provided = %s",str_thread_support(required),str_thread_support(provided));

  // Call die instead of throwing exceptions from OTHER_ASSERT, OTHER_NOT_IMPLEMENTED, and THROW.
  set_error_callback(static_cast<ErrorCallback>(mpi_die_helper));
  die_callback = mpi_die_helper;
  throw_callback = mpi_die_helper;

  // Make MPI errors return so that we can check the error codes ourselves
  MPI_Comm_set_errhandler(MPI_COMM_WORLD,MPI_ERRORS_RETURN);
}

mpi_world_t::~mpi_world_t() {
  CHECK(MPI_Finalize());
}

}
}
