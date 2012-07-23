// Parallel I/O for slice files
#pragma once

#include <pentago/mpi/block_store.h>
#include <mpi.h>
namespace pentago {
namespace mpi {

// Write all data to a single slice file.  Collective.
void write_sections(const MPI_Comm comm, const string& filename, const block_store_t& blocks, const int level);

void check_directory(MPI_Comm comm, const string& dir);

}
}
