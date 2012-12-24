// Include mpi.h without getting deprecated C++ bindings
#pragma once

#ifdef OMPI_MPI_H
#error "mpi.h" already included
#endif

#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

// Uncomment to enable expensive consistency checking
//#define PENTAGO_MPI_DEBUG

// Uncomment to enable MPI tracing
//#define PENTAGO_MPI_TRACING

// Whether or not to store blocks compressed
#define PENTAGO_MPI_COMPRESS 1

// Whether or not to use interleave filtered to precondition snappy
#define PENTAGO_MPI_SNAPPY_FILTER 1

namespace pentago {

// We fix the block size at compile time for optimization reasons
const int block_size = 8;
const int block_shift = 3;

// Hopefully conservative estimate of snappy's compression ratio on our data
const double snappy_compression_estimate = .5;

// Whether or not to merge matching block requests
const bool merge_block_requests = true;

}
