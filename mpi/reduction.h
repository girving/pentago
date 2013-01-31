// Wrappers around MPI reductions to isolate MPI dependencies
#pragma once

#include <pentago/end/config.h>
#include <pentago/end/reduction.h>
#include <pentago/mpi/utility.h>
#include <other/core/utility/curry.h>
#include <mpi.h>
namespace pentago {
namespace mpi {

// Map from C++ types to MPI types
template<class T> static inline MPI_Datatype datatype();
template<> inline MPI_Datatype datatype<double>() { return MPI_DOUBLE; }
template<> inline MPI_Datatype datatype<long>()   { return MPI_LONG; }

// Map from op_t to MPI_Op
template<op_t op> static inline MPI_Op mpi_op();
template<> inline MPI_Op mpi_op<sum_op>() { return MPI_SUM; }
template<> inline MPI_Op mpi_op<max_op>() { return MPI_MAX; }

template<class T,op_t op> static bool reduce(const MPI_Comm comm, RawArray<T> buffer) {
  const int rank = comm_rank(comm);
  CHECK(MPI_Reduce(rank?buffer.data():MPI_IN_PLACE,buffer.data(),buffer.size(),datatype<T>(),mpi_op<op>(),0,comm));
  return !rank;
}

template<class T,op_t op> static inline reduction_t<T,op> reduction(const MPI_Comm comm) {
  return reduction_t<T,op>(curry(reduce<T,op>,comm));
}

}
}
