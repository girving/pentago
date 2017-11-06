// MPI related utilities

#include "pentago/mpi/utility.h"
#include "pentago/mpi/trace.h"
#include "pentago/utility/ceil_div.h"
#include "pentago/utility/debug.h"
namespace pentago {
namespace mpi {

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

static void mpi_die_helper(const string& msg) __attribute__((noreturn, cold));
static void mpi_die_helper(const string& msg) {
  std::cerr << "\nrank " << comm_rank(MPI_COMM_WORLD) << ": " << msg << std::endl;
  if (getenv("GEODE_BREAK_ON_ASSERT"))
    breakpoint();
  // Ideally we would call MPI_Abort here, but that's technically disallowed if we're not
  // running in MPI_THREAD_MULTIPLE mode.  Therefore, we bail more forcefully.
  abort();
}

#if defined(OPEN_MPI) && (OMPI_MAJOR_VERSION>1 || (OMPI_MAJOR_VERSION==1 && OMPI_MINOR_VERSION>=4))
// Recent openmpi claims to only support single threaded when compiled with default options,
// as used by stock Ubuntu.  However, it appears to actually supported funneled, as indicated
// by http://www.open-mpi.org/community/lists/users/2011/05/16451.php.  Therefore, we lie.
static const int funneled = MPI_THREAD_SINGLE;
#else
static const int funneled = MPI_THREAD_FUNNELED;
#endif

mpi_world_t::mpi_world_t(int& argc, char**& argv) {
  const int required = PENTAGO_MPI_FUNNEL ? funneled : MPI_THREAD_MULTIPLE;
  int provided;
  CHECK(MPI_Init_thread(&argc,&argv,required,&provided));
  if (provided<required)
    die("Insufficent MPI thread support: required = %s, provided = %s",str_thread_support(required),str_thread_support(provided));

  // Record rank for tracing purposes
  set_mpi_trace_rank(comm_rank(MPI_COMM_WORLD));

  // Call die instead of throwing exceptions from GEODE_ASSERT, GEODE_NOT_IMPLEMENTED, and THROW.
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
