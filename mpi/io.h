// Parallel I/O for slice files
#pragma once

#include <pentago/mpi/block_store.h>
#include <mpi.h>
namespace pentago {
namespace mpi {

// Write all data to a single slice file.  Collective.
void write_sections(const MPI_Comm comm, const string& filename, const block_store_t& blocks, const int level);

// Write an empty section file to the directory to check that basic I/O works
void check_directory(const MPI_Comm comm, const string& dir);

// Write per-section counts to a .npy file.  Collective.
// The numpy array is a sequence of (section,black-win-counts,white-win-counts,total-counts) tuples packed as 4 uint64_t's.
void write_counts(const MPI_Comm comm, const string& filename, const block_store_t& blocks);

// Write sparse samples to a .npy file.  Collective.
// The format is a sequence of (board,black-wins,white-wins) tuples packed as 9 uint64_t's.
// The samples chosen are identical to those chosen by the out-of-core solver, but in somewhat scrambled order.
void write_sparse_samples(const MPI_Comm, const string& filename, const block_store_t& blocks, const int samples_per_section);

}
}
