// Parallel I/O for slice files

#include <pentago/mpi/io.h>
#include <pentago/mpi/utility.h>
#include <pentago/filter.h>
#include <pentago/compress.h>
#include <pentago/supertensor.h>
#include <pentago/utility/aligned.h>
#include <pentago/utility/char_view.h>
#include <pentago/utility/memory.h>
#include <other/core/python/numpy.h>
#include <other/core/random/Random.h>
#include <other/core/structure/HashtableIterator.h>
#include <other/core/utility/curry.h>
#include <other/core/utility/ProgressIndicator.h>
#include <other/core/utility/Hasher.h>
#include <other/core/utility/Log.h>
#include <tr1/unordered_map>
namespace pentago {
namespace mpi {

using Log::cout;
using std::endl;
using std::make_pair;
using std::tr1::unordered_map;

/* Notes:
 *
 * 1. In order to use MPI_File_write_all collectives for the actual writing, blocks and sections will
 *    be written out in only loosely ordered form based on the order of ranks.  This does roughly correspond
 *    to natural ordering (at least for now) due to the way we partition sections and lines.
 *
 * 2. A processor is considered to own a section if it owns the (0,0,0,0) block of that section.  The owner
 *    of a section is responsible for writing out the block index of that section.  As in single section
 *    supertensor writing, block indices are written last so that they can be compressed without affecting
 *    the positions of blocks.
 *
 * 3. Rank 0 is responsible for writing out the headers.  These go at the beginning of the file, and are
 *    uncompressed.  See supertensor.h for the format.
 *
 * 4. For now, we hard code interleave filtering (filter = 1)
 */

static void compress_and_store(Array<uint8_t>* dst, RawArray<const uint8_t> data, int level) {
  *dst = compress(data,level);
}

#if PENTAGO_MPI_COMPRESS
static void filter_and_compress_and_store(Tuple<spinlock_t,ProgressIndicator>* progress, Array<uint8_t>* dst, const block_store_t* blocks, int local_id, int level, bool turn) {
#else
static void filter_and_compress_and_store(Tuple<spinlock_t,ProgressIndicator>* progress, Array<uint8_t>* dst, RawArray<const Vector<super_t,2>> data, int level, bool turn) {
#endif

#if PENTAGO_MPI_COMPRESS
  // Uncompress if necessary
  Array<Vector<super_t,2>> data = blocks->uncompress_and_get_flat(local_id);
#endif

  // Adjust for the different format of block_store_t and apply interleave filtering
#if PENTAGO_MPI_COMPRESS
  const auto filtered = data.raw();
#else
  Array<Vector<super_t,2>> filtered;
#endif
  {
    thread_time_t time(filter_kind);
#if !PENTAGO_MPI_COMPRESS
    filtered = large_buffer<Vector<super_t,2>>(data.size(),false);
#endif
    if (!turn)
      for (int i=0;i<data.size();i++)
        filtered[i] = interleave_super(vec(data[i].x,~data[i].y));
    else
      for (int i=0;i<data.size();i++)
        filtered[i] = interleave_super(vec(~data[i].y,data[i].x));
  }
  // Compress
  *dst = compress(char_view(filtered),level);
  // Tick
  spin_t spin(progress->x); 
  progress->y.progress();
}

static void file_open(const MPI_Comm comm, const string& filename, MPI_File* file) {
  const int amode = MPI_MODE_WRONLY | MPI_MODE_CREATE;
  int r = MPI_File_open(comm,(char*)filename.c_str(),amode,MPI_INFO_NULL,file);
  if (r != MPI_SUCCESS)
    die(format("failed to open '%s' for writing: %s",filename,error_string(r)));
}

void write_sections(const MPI_Comm comm, const string& filename, const block_store_t& blocks, const int level) {
  const int ranks = comm_size(comm),
            rank = comm_rank(comm);
  const auto& partition = *blocks.partition;
  const int filter = 1; // interleave filtering
  const bool turn = partition.slice&1;

  // Open the file
  MPI_File file;
  {
    thread_time_t time(write_sections_kind);
    file_open(comm,filename,&file);
  }

  // Compress all local data
  const int local_blocks = blocks.blocks();
  vector<Array<uint8_t>> compressed(local_blocks);
  auto progress = tuple(spinlock_t(),ProgressIndicator(local_blocks));
  for (int b : range(local_blocks))
#if PENTAGO_MPI_COMPRESS
    threads_schedule(CPU,curry(filter_and_compress_and_store,&progress,&compressed[b],&blocks,b,level,turn));
#else
    threads_schedule(CPU,curry(filter_and_compress_and_store,&progress,&compressed[b],blocks.get_raw_flat(b),level,turn));
#endif
  threads_wait_all_help();

  // Determine the base offset of each rank
  const auto sections = partition.sections.raw();
  const auto& section_id = partition.section_id;
  const uint64_t header_size = supertensor_magic_size+3*sizeof(uint32_t)+supertensor_header_t::header_size*sections.size();
  uint64_t local_size = 0;
  uint64_t previous = 0;
  {
    thread_time_t time(write_sections_kind);
    for (const auto& c : compressed)
      local_size += c.size();
    OTHER_ASSERT(local_size<(1u<<31));
    CHECK(MPI_Exscan(&local_size,&previous,1,MPI_LONG_LONG_INT,MPI_SUM,comm));
  }

  // Broadcast offset of first block index to everyone
  uint64_t block_index_start = rank==ranks-1?header_size+previous+local_size:0;
  {
    thread_time_t time(write_sections_kind);
    CHECK(MPI_Bcast(&block_index_start,1,MPI_LONG_LONG_INT,ranks-1,comm));
  }

  // Compute local block offsets
  Array<supertensor_blob_t> block_blobs(local_blocks+1,false);
  block_blobs[0].offset = header_size+previous;
  {
    thread_time_t time(write_sections_kind);
    for (int b : range(local_blocks)) {
      const auto info = blocks.block_info[b];
      block_blobs[b].compressed_size = compressed[b].size();
      block_blobs[b].uncompressed_size = sizeof(Vector<super_t,2>)*block_shape(info.section.shape(),info.block).product();
      block_blobs[b+1].offset = block_blobs[b].offset+compressed[b].size();
    }
    OTHER_ASSERT(block_blobs.last().offset==header_size+previous+local_size);
  }

  // Concatenate local data into a single buffer
  {
    thread_time_t time(write_sections_kind);
    Array<uint8_t> buffer(local_size,false);
    int next = 0;
    for (const auto& c : compressed) {
      memcpy(buffer.data()+next,c.data(),c.size());
      next += c.size();
    }
    OTHER_ASSERT(next==buffer.size());
    compressed.clear();

    // Write all blocks, skipping space at the beginning of the file for headers
    CHECK(MPI_File_write_at_all(file,header_size+previous,buffer.data(),buffer.size(),MPI_BYTE,MPI_STATUS_IGNORE));
  }

  // Now we need to write the block indexes, which requires sending all block offsets in a given section
  // to one processor for compression and writing.  We take advantage of the fact that in the current
  // partitioning scheme each rank owns blocks from at most a few sections, so it is reasonably efficient
  // to send one message per rank per section.  Each rank organizes its block offsets by section, does
  // a bunch of MPI_Isends, and then waits to receive information about any sections it owns.

  Hashtable<int,Array<supertensor_blob_t,4>> block_indexes;
  {
    thread_time_t time(write_sections_kind);

    // Organize block information by section
    int remaining_owned_blocks = 0;
    typedef Tuple<Vector<int,4>,supertensor_blob_t> block_blob_t;
    BOOST_STATIC_ASSERT(sizeof(block_blob_t)==10*sizeof(int));
    Hashtable<int,Array<block_blob_t>> local_block_blobs;
    for (int b : range(local_blocks)) {
      const auto info = blocks.block_info[b];
      const int sid = section_id.get(info.section);
      if (info.block==Vector<int,4>()) {
        const auto section_blocks = pentago::mpi::section_blocks(sections[sid]);
        Array<supertensor_blob_t,4> block_index(section_blocks,false);
        memset(block_index.data(),0,sizeof(supertensor_blob_t)*block_index.flat.size());
        block_indexes.set(sid,block_index);
        remaining_owned_blocks += section_blocks.product();
      }
      auto& blobs = local_block_blobs.get_or_insert(sid);
      blobs.append(tuple(info.block,block_blobs[b]));
    }

    // Send our size information to the appropriate processor
    Array<MPI_Request> requests;
    for (HashtableIterator<int,Array<block_blob_t>> it(local_block_blobs);it.valid();it.next()) {
      const int sid = it.key();
      RawArray<const block_blob_t> blobs = it.data();
      const int rank = partition.block_to_rank(sections[sid],Vector<int,4>());
      MPI_Request request; 
      // Use the section id as the tag for identification purposes 
      CHECK(MPI_Isend((void*)blobs.data(),sizeof(block_blob_t)/sizeof(int)*blobs.size(),MPI_INT,rank,sid,comm,&request));
      requests.append(request);
    }

    // Receive size information for all sections that we own
    while (remaining_owned_blocks) {
      // Probe for size messages and receive them
      MPI_Status status;
      CHECK(MPI_Probe(MPI_ANY_SOURCE,MPI_ANY_TAG,comm,&status));
      const int sid = status.MPI_TAG;
      const int count = get_count(&status,MPI_INT);
      OTHER_ASSERT(count%(sizeof(block_blob_t)/sizeof(int))==0);
      Array<block_blob_t> blobs(count/(sizeof(block_blob_t)/sizeof(int)),false);
      CHECK(MPI_Recv(blobs.data(),count,MPI_INT,status.MPI_SOURCE,sid,comm,MPI_STATUS_IGNORE));

      // Absorb the size information
      remaining_owned_blocks -= blobs.size();
      Array<supertensor_blob_t,4>& block_index = block_indexes.get(sid);
      for (const auto& blob : blobs) {
        OTHER_ASSERT(block_index.valid(blob.x));
        OTHER_ASSERT(!block_index[blob.x].offset);
        block_index[blob.x] = blob.y;
      }
    }

    // Wait for all pending requests
    CHECK(MPI_Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
  }

  // Compress all block indexes
  thread_time_t time(write_sections_kind);
  vector<Tuple<int,Array<uint8_t>>> compressed_block_indexes;
  compressed_block_indexes.reserve(block_indexes.size());
  for (HashtableIterator<int,Array<supertensor_blob_t,4>> it(block_indexes);it.valid();it.next()) {
    compressed_block_indexes.push_back(tuple(it.key(),Array<uint8_t>()));
    threads_schedule(CPU,curry(compress_and_store,&compressed_block_indexes.back().y,char_view(it.data().flat),level));
  }
  threads_wait_all_help();
  Hashtable<int,Array<supertensor_blob_t,4>>().swap(block_indexes); // Deallocate
  #define block_indexes hide_block_indexes

  // Compute block index offsets
  uint64_t local_block_indexes_size = 0;
  for (const auto& bi : compressed_block_indexes)
    local_block_indexes_size += bi.y.size();
  OTHER_ASSERT(local_block_indexes_size<(1u<<31));
  uint64_t previous_block_indexes_size = 0;
  CHECK(MPI_Exscan(&local_block_indexes_size,&previous_block_indexes_size,1,MPI_LONG_LONG_INT,MPI_SUM,comm));
  const uint64_t local_block_index_start = block_index_start+previous_block_indexes_size;

  // Send all block index blobs to root
  Array<supertensor_blob_t> index_blobs(sections.size(),false);
  memset(index_blobs.data(),0,sizeof(supertensor_blob_t)*sections.size());
  int next_block_index_offset = local_block_index_start;
  for (const auto& bi : compressed_block_indexes) {
    auto& blob = index_blobs[bi.x];
    blob.uncompressed_size = sizeof(supertensor_blob_t)*section_blocks(sections[bi.x]).product();
    blob.compressed_size = bi.y.size();
    blob.offset = next_block_index_offset;
    next_block_index_offset += bi.y.size();
  }
  CHECK(MPI_Reduce(rank?index_blobs.data():MPI_IN_PLACE,index_blobs.data(),sizeof(supertensor_blob_t)/sizeof(uint64_t)*sections.size(),MPI_LONG_LONG_INT,MPI_SUM,0,comm));
  if (rank)
    index_blobs = Array<supertensor_blob_t>();

  // Concatenate compressed block indexes into one buffer
  Array<uint8_t> all_block_indexes(local_block_indexes_size,false);
  uint64_t next_block_index = 0;
  for (const auto& bi : compressed_block_indexes) {
    memcpy(all_block_indexes.data()+next_block_index,bi.y.data(),bi.y.size());
    next_block_index += bi.y.size();
  }

  // Write all block indexes
  CHECK(MPI_File_write_at_all(file,local_block_index_start,all_block_indexes.data(),all_block_indexes.size(),MPI_BYTE,MPI_STATUS_IGNORE));
  all_block_indexes = Array<uint8_t>(); // Deallocate
  #define all_block_indexes hide_all_block_indexes
 
  // On rank 0, write all section headers
  if (!rank) {
    Array<uint8_t> headers(header_size,false);
    size_t offset = 0;
    #define HEADER(pointer,size) \
      memcpy(headers.data()+offset,pointer,size); \
      offset += size;
    const uint32_t version = 3;
    const uint32_t section_count = sections.size();
    const uint32_t section_header_size = supertensor_header_t::header_size;
    HEADER(multiple_supertensor_magic,supertensor_magic_size);
    HEADER(&version,sizeof(uint32_t))
    HEADER(&section_count,sizeof(uint32_t))
    HEADER(&section_header_size,sizeof(uint32_t))
    for (int s=0;s<sections.size();s++) {
      supertensor_header_t sh(sections[s],block_size,filter);
      sh.valid = true;
      sh.index = index_blobs[s];
      sh.pack(headers.slice(offset+range(sh.header_size)));
      offset += sh.header_size;
    }
    OTHER_ASSERT(offset==header_size);
    // Write the last piece of the file
    CHECK(MPI_File_write_at(file,0,headers.data(),headers.size(),MPI_BYTE,MPI_STATUS_IGNORE));
  }

  // Done!
  CHECK(MPI_File_close(&file));
}

void check_directory(const MPI_Comm comm, const string& dir) {
  const int slice = 24;
  const int rank = comm_rank(comm);
  const int samples_per_section = 0;
  const Array<const section_t> sections;
  const auto partition = new_<partition_t>(comm_size(comm),slice,sections);
  const auto lines = partition->rank_lines(rank,true);
  const auto blocks = new_<block_store_t>(partition,rank,samples_per_section,lines);
  write_sections(comm,format("%s/empty.pentago",dir),blocks,0);
}

void write_counts(const MPI_Comm comm, const string& filename, const block_store_t& blocks) {
  thread_time_t time(write_counts_kind);

  // Reduce win counts down to root, destroying them in the process
  const int rank = comm_rank(comm);
  const RawArray<Vector<uint64_t,3>> counts = blocks.section_counts;
  if (rank) {
    CHECK(MPI_Reduce(counts.data(),0,3*counts.size(),MPI_LONG_LONG_INT,MPI_SUM,0,comm));
    // Only the root does the actual writing
    return;
  }
  // From here on we're the root
  CHECK(MPI_Reduce(MPI_IN_PLACE,counts.data(),3*counts.size(),MPI_LONG_LONG_INT,MPI_SUM,0,comm));

  // Prepare data array
  Array<Vector<uint64_t,4>> data(counts.size(),false);
  const bool turn = blocks.partition->slice&1;
  for (int i=0;i<data.size();i++) {
    auto wins = counts[i].x, losses = counts[i].z-counts[i].y;
    if (turn)
      swap(wins,losses);
    data[i].set(blocks.partition->sections[i].sig(),wins,losses,counts[i].z);
  }

  // Pack numpy buffer
  Array<uint8_t> buffer(256+memory_usage(data),false);
  size_t data_size = fill_numpy_header(buffer,data);
  OTHER_ASSERT(data_size==sizeof(Vector<uint64_t,4>)*data.size()); 
  int header_size = buffer.size();
  buffer.resize(header_size+data_size,false,true);
  memcpy(buffer.data()+header_size,data.data(),data_size);

  // Write file
  MPI_File file;
  file_open(MPI_COMM_SELF,filename,&file);
  CHECK(MPI_File_write(file,buffer.data(),buffer.size(),MPI_BYTE,MPI_STATUS_IGNORE));
  CHECK(MPI_File_close(&file));
}

void write_sparse_samples(const MPI_Comm comm, const string& filename, block_store_t& blocks) {
  thread_time_t time(write_sparse_kind);
  const int rank = comm_rank(comm);
  const bool turn = blocks.partition->slice&1;

  // Mangle samples into correct output format in place
  typedef block_store_t::sample_t sample_t;
  const RawArray<sample_t> samples = blocks.samples.flat;
  if (!turn) // Black to play
    for (auto& sample : samples)
      sample.wins.y = ~sample.wins.y;
  else // White to play
    for (auto& sample : samples) {
      sample.wins.y = ~sample.wins.y;
      swap(sample.wins.x,sample.wins.y);
    }

  // Generate a datatype for the pieces of block_store_t::sample_t that we need
  MPI_Datatype sample_type;
  int lengths[2] = {1,8};
  MPI_Aint displacements[2] = {offsetof(sample_t,board),offsetof(sample_t,wins)};
  MPI_Datatype types[2] = {MPI_LONG_LONG_INT,MPI_LONG_LONG_INT};
  CHECK(MPI_Type_create_struct(2,lengths,displacements,types,&sample_type));

  // Count total samples and send to root
  const int local_samples = samples.size();
  int total_samples;
  CHECK(MPI_Reduce((void*)&local_samples,&total_samples,1,MPI_INT,MPI_SUM,0,comm));

  // Open the file
  MPI_File file;
  file_open(comm,filename,&file);

  // Write our data
  if (rank) {
    // On non-root processes, we only need to write out samples
    CHECK(MPI_Type_commit(&sample_type));
    CHECK(MPI_File_write_ordered(file,samples.data(),samples.size(),sample_type,MPI_STATUS_IGNORE));
  } else {
    // On the root, we have to write out both header and our samples.  First, build the header.
    Array<uint8_t> header;
    {
      RawArray<Vector<uint64_t,9>> all_samples(total_samples,0); // False array of all samples
      fill_numpy_header(header,all_samples);
    }
    // Construct datatype combining header with local samples
    int lengths[2] = {header.size(),samples.size()};
    MPI_Aint displacements[2] = {(MPI_Aint)header.data(),(MPI_Aint)samples.data()};
    MPI_Datatype types[2] = {MPI_BYTE,sample_type};
    MPI_Datatype datatype;
    CHECK(MPI_Type_struct(2,lengths,displacements,types,&datatype));
    CHECK(MPI_Type_commit(&datatype));
    // Write 
    CHECK(MPI_File_write_ordered(file,0,1,datatype,MPI_STATUS_IGNORE));
    // Free datatype
    CHECK(MPI_Type_free(&datatype));
  }

  // Done!
  CHECK(MPI_File_close(&file));
  CHECK(MPI_Type_free(&sample_type));
}

}
}
