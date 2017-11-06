// Parallel I/O for slice files
//
// These routines duplicate the functionality of supertensor.{h,cpp},
// but in a massively parallel setting using MPI I/O.
#pragma once

#include "pentago/end/block_store.h"
#include "pentago/end/config.h"
#include "pentago/end/partition.h"
#include <mpi.h>
namespace pentago {
namespace mpi {

using namespace pentago::end;

// Write all data to a single slice file.  Collective.
void write_sections(const MPI_Comm comm, const string& filename, const readable_block_store_t& blocks,
                    const int level);

// Read all data from a set of supertensor files
shared_ptr<const readable_block_store_t> read_sections(
    const MPI_Comm comm, const string& filename, const shared_ptr<compacting_store_t>& store,
    const partition_factory_t& partition_factory);

// Consistency check for some of the read sections code
void read_sections_test(const MPI_Comm comm, const string& filename,
                        const partition_factory_t& partition_factory);

// Write an empty section file to the directory to check that basic I/O works
void check_directory(const MPI_Comm comm, const string& dir);

// Write per-section counts to a .npy file.  Collective.
// The numpy array is a sequence of (section,black-win-counts,white-win-counts,total-counts) tuples
// packed as 4 uint64_t's.
void write_counts(const MPI_Comm comm, const string& filename, const accumulating_block_store_t& blocks);

// Write sparse samples to a .npy file.  Collective.
// The format is a sequence of (board,black-wins,white-wins) tuples packed as 9 uint64_t's.
// The samples chosen are identical to those chosen by the out-of-core solver, but in somewhat scrambled
// order.
//
// Update: samples are now collected as blocks complete inside accumulating_block_store_t in order to
// avoid an unnecessary decompression pass.
void write_sparse_samples(const MPI_Comm, const string& filename, accumulating_block_store_t& blocks);

}
}
