// Parallel I/O for slice files

#include <pentago/mpi/io.h>
#include <pentago/mpi/utility.h>
#include <pentago/end/blocks.h>
#include <pentago/data/filter.h>
#include <pentago/data/compress.h>
#include <pentago/data/supertensor.h>
#include <pentago/utility/aligned.h>
#include <pentago/utility/debug.h>
#include <pentago/utility/char_view.h>
#include <pentago/utility/endian.h>
#include <pentago/utility/index.h>
#include <pentago/utility/memory.h>
#include <other/core/python/numpy.h>
#include <other/core/random/Random.h>
#include <other/core/structure/Hashtable.h>
#include <other/core/utility/curry.h>
#include <other/core/utility/ProgressIndicator.h>
#include <other/core/utility/Hasher.h>
#include <other/core/utility/Log.h>
#include <other/core/utility/openmp.h>
namespace pentago {
namespace mpi {

using Log::cout;
using std::endl;
using std::make_pair;
using namespace pentago::mpi;

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

static void compress_and_store(Array<uint8_t>* dst, RawArray<supertensor_blob_t> data, int level) {
  to_little_endian_inplace(data);
  *dst = compress(char_view(data),level,unevent);
}

#if PENTAGO_MPI_COMPRESS
static void filter_and_compress_and_store(Tuple<spinlock_t,ProgressIndicator>* progress, Array<uint8_t>* dst, const block_store_t* blocks, local_id_t local_id, int level, bool turn) {
#else
static void filter_and_compress_and_store(Tuple<spinlock_t,ProgressIndicator>* progress, Array<uint8_t>* dst, RawArray<const Vector<super_t,2>> data, int level, bool turn) {
#endif

#if PENTAGO_MPI_COMPRESS
  // Uncompress if necessary
  const event_t event = blocks->local_block_event(local_id);
  const auto data = blocks->uncompress_and_get_flat(local_id,event);
#else
  const event_t event = unevent;
#endif

  // Adjust for the different format of block_store_t and apply interleave filtering
#if PENTAGO_MPI_COMPRESS
  const auto filtered = data;
#else
  Array<Vector<super_t,2>> filtered;
#endif
  {
    thread_time_t time(filter_kind,event);
#if !PENTAGO_MPI_COMPRESS
    filtered = large_buffer<Vector<super_t,2>>(data.size(),false);
#endif
    if (!turn)
      for (int i=0;i<data.size();i++)
        filtered[i] = to_little_endian(interleave_super(vec(data[i].x,~data[i].y)));
    else
      for (int i=0;i<data.size();i++)
        filtered[i] = to_little_endian(interleave_super(vec(~data[i].y,data[i].x)));
  }
  // Compress
  *dst = compress(char_view(filtered),level,event);
  // Tick
  spin_t spin(progress->x); 
  progress->y.progress();
}

static void file_open(const MPI_Comm comm, const string& filename, MPI_File* file) {
  const int amode = MPI_MODE_WRONLY | MPI_MODE_CREATE;
  int r = MPI_File_open(comm,(char*)filename.c_str(),amode,MPI_INFO_NULL,file);
  if (r != MPI_SUCCESS)
    die("failed to open '%s' for writing: %s",filename,error_string(r));
}

void write_sections(const MPI_Comm comm, const string& filename, const block_store_t& blocks, const int level) {
  const int ranks = comm_size(comm),
            rank = comm_rank(comm);
  const auto& partition = *blocks.partition;
  const int filter = 1; // interleave filtering
  const bool turn = partition.sections->slice&1;
  const event_t event = unevent;

  // Open the file
  MPI_File file;
  {
    thread_time_t time(write_sections_kind,event);
    file_open(comm,filename,&file);
  }

  // Compress all local data
  const int local_blocks = blocks.total_blocks();
  Array<const block_info_t*> flat_info(local_blocks); // Allow indexing by flat_id
  vector<Array<uint8_t>> compressed(local_blocks);
  auto progress = tuple(spinlock_t(),ProgressIndicator(local_blocks));
  for (const auto& info : blocks.block_infos) {
    flat_info[info.data.flat_id] = &info.data;
#if PENTAGO_MPI_COMPRESS
    threads_schedule(CPU,curry(filter_and_compress_and_store,&progress,&compressed[info.data.flat_id],&blocks,info.key,level,turn));
#else
    threads_schedule(CPU,curry(filter_and_compress_and_store,&progress,&compressed[info.data.flat_id],blocks.get_raw_flat(info.key),level,turn));
#endif
  }
  threads_wait_all_help();

  // Determine the base offset of each rank
  const auto sections = partition.sections->sections.raw();
  const auto& section_id = partition.sections->section_id;
  const uint64_t header_size = supertensor_magic_size+3*sizeof(uint32_t)+supertensor_header_t::header_size*sections.size();
  uint64_t local_size = 0;
  uint64_t previous = 0;
  {
    thread_time_t time(write_sections_kind,event);
    for (const auto& c : compressed)
      local_size += c.size();
    OTHER_ASSERT(local_size<(1u<<31));
    CHECK(MPI_Exscan(&local_size,&previous,1,datatype<uint64_t>(),MPI_SUM,comm));
  }

  // Broadcast offset of first block index to everyone
  uint64_t block_index_start = rank==ranks-1?header_size+previous+local_size:0;
  {
    thread_time_t time(write_sections_kind,event);
    CHECK(MPI_Bcast(&block_index_start,1,datatype<uint64_t>(),ranks-1,comm));
  }

  // Compute local block offsets
  Array<supertensor_blob_t> block_blobs(local_blocks+1,false);
  block_blobs[0].offset = header_size+previous;
  {
    thread_time_t time(write_sections_kind,event);
    for (const int b : range(local_blocks)) {
      const block_info_t& info = *flat_info[b];
      block_blobs[b].uncompressed_size = sizeof(Vector<super_t,2>)*block_shape(info.section.shape(),info.block).product();
      block_blobs[b].compressed_size = compressed[b].size();
      block_blobs[b+1].offset = block_blobs[b].offset+compressed[b].size();
    }
    OTHER_ASSERT(block_blobs.back().offset==header_size+previous+local_size);
  }

  // Concatenate local data into a single buffer
  {
    thread_time_t time(write_sections_kind,event);
    Array<uint8_t> buffer(local_size,false);
    int next = 0;
    for (auto& c : compressed) {
      memcpy(buffer.data()+next,c.data(),c.size());
      next += c.size();
      c.clean_memory();
    }
    OTHER_ASSERT(next==buffer.size());
    vector<Array<uint8_t>>().swap(compressed);

    // Write all blocks, skipping space at the beginning of the file for headers
    CHECK(MPI_File_write_at_all(file,header_size+previous,buffer.data(),buffer.size(),MPI_BYTE,MPI_STATUS_IGNORE));
  }

  // Now we need to write the block indexes, which requires sending all block offsets in a given section
  // to one processor for compression and writing.  
  //
  // We used to take advantage of the fact that in the simple partitioning scheme each rank owns blocks
  // from at most a few sections, so it is reasonably efficient to send one message per rank per section.
  // Each rank organized its block offsets by section, did a bunch of MPI_Isends, and then waited to
  // receive information about any sections it owns.
  //
  // Now, however, we have the option of scattering blocks randomly amongst the processors.  For simplicity,
  // we assign section ownership using section ids and partition_loop, then use MPI_Alltoallv to send all
  // block blobs to the appropriate section owner.

  const Range<int> section_range = partition_loop(sections.size(),ranks,rank);
  vector<Array<supertensor_blob_t,4>> block_indexes(section_range.size()); // Section s is stored in entry s-section_range.lo
  {
    thread_time_t time(write_sections_kind,event);

    // Count the number of blobs to send to each process
    const Array<int> send_counts(ranks);
    for (const int b : range(local_blocks)) {
      const auto& info = *flat_info[b]; 
      const int sid = section_id.get(info.section);
      send_counts[partition_loop_inverse(sections.size(),ranks,sid)]++;
    }

    // Build an MPI datatype for block,blob pairs
    typedef Tuple<int,Vector<uint8_t,4>,supertensor_blob_t> block_blob_t; // section_id, block, blob
    BOOST_STATIC_ASSERT(sizeof(block_blob_t)==8*sizeof(int));
    MPI_Datatype block_blob_datatype;
    CHECK(MPI_Type_contiguous(sizeof(block_blob_t)/sizeof(int),MPI_INT,&block_blob_datatype));
    CHECK(MPI_Type_commit(&block_blob_datatype));

    // Construct send buffer
    const NestedArray<block_blob_t> send_buffer(send_counts,false);
    const auto remaining = send_counts.copy();
    for (const int b : range(local_blocks)) {
      const auto& info = *flat_info[b]; 
      const int sid = section_id.get(info.section);
      const int rank = partition_loop_inverse(sections.size(),ranks,sid);
      send_buffer(rank,--remaining[rank]) = tuple(sid,info.block,block_blobs[b]);
    }
    flat_info.clean_memory();
    #define flat_info hide_flat_info

    // Allocate block indexes for each section we own, and determine how many blobs will come from where
    const Array<int> recv_counts(ranks);
    for (const int sid : section_range) {
      const auto section_blocks = pentago::mpi::section_blocks(sections[sid]);
      Array<supertensor_blob_t,4> block_index(section_blocks,false);
      memset(block_index.data(),0,sizeof(supertensor_blob_t)*block_index.flat.size());
      block_indexes[sid-section_range.lo] = block_index;
      for (const int i : range(section_blocks.product())) {
        const Vector<uint8_t,4> block(decompose(section_blocks,i));
        recv_counts[partition.find_block(sections[sid],block).x]++;
      }
    }

    // Communicate
    const NestedArray<block_blob_t> recv_buffer(recv_counts,false);
    CHECK(MPI_Alltoallv(send_buffer.flat.data(),send_counts.data(),send_buffer.offsets.const_cast_().data(),block_blob_datatype,
                        recv_buffer.flat.data(),recv_counts.data(),recv_buffer.offsets.const_cast_().data(),block_blob_datatype,comm));
    CHECK(MPI_Type_free(&block_blob_datatype));

    // Packed received blobs into block indexes
    for (const auto& block_blob : recv_buffer.flat) {
      const int sid = block_blob.x;
      OTHER_ASSERT(section_range.contains(sid));
      block_indexes[sid-section_range.lo][Vector<int,4>(block_blob.y)] = block_blob.z;
    }
  }

  // Verify that all blobs are initialized
  for (const auto& block_index : block_indexes)
    for (const auto& blob : block_index.flat)
      OTHER_ASSERT(blob.uncompressed_size);

  // Compress all block indexes
  vector<Array<uint8_t>> compressed_block_indexes(section_range.size());
  for (const int sid : section_range) {
    compressed_block_indexes[sid-section_range.lo] = Array<uint8_t>();
    threads_schedule(CPU,curry(compress_and_store,&compressed_block_indexes[sid-section_range.lo],block_indexes[sid-section_range.lo].flat,level));
  }
  threads_wait_all_help();
  vector<Array<supertensor_blob_t,4>>().swap(block_indexes);
  #define block_indexes hide_block_indexes

  // Compute block index offsets
  thread_time_t time(write_sections_kind,event);
  uint64_t local_block_indexes_size = 0;
  for (const int sid : section_range)
    local_block_indexes_size += compressed_block_indexes[sid-section_range.lo].size();
  OTHER_ASSERT(local_block_indexes_size<(1u<<31));
  uint64_t previous_block_indexes_size = 0;
  CHECK(MPI_Exscan(&local_block_indexes_size,&previous_block_indexes_size,1,datatype<uint64_t>(),MPI_SUM,comm));
  const uint64_t local_block_index_start = block_index_start+previous_block_indexes_size;

  // Send all block index blobs to root
  Array<supertensor_blob_t> index_blobs(sections.size(),false);
  memset(index_blobs.data(),0,sizeof(supertensor_blob_t)*sections.size());
  int next_block_index_offset = local_block_index_start;
  for (const int sid : section_range) {
    auto& blob = index_blobs[sid];
    blob.uncompressed_size = sizeof(supertensor_blob_t)*section_blocks(sections[sid]).product();
    blob.compressed_size = compressed_block_indexes[sid-section_range.lo].size();
    blob.offset = next_block_index_offset;
    next_block_index_offset += blob.compressed_size;
  }
  CHECK(MPI_Reduce(rank?index_blobs.data():MPI_IN_PLACE,index_blobs.data(),sizeof(supertensor_blob_t)/sizeof(uint64_t)*sections.size(),datatype<uint64_t>(),MPI_SUM,0,comm));
  if (rank)
    index_blobs = Array<supertensor_blob_t>();

  // Concatenate compressed block indexes into one buffer
  Array<uint8_t> all_block_indexes(local_block_indexes_size,false);
  uint64_t next_block_index = 0;
  for (const int sid : section_range) {
    const auto block_index = compressed_block_indexes[sid-section_range.lo].raw();
    memcpy(all_block_indexes.data()+next_block_index,block_index.data(),block_index.size());
    next_block_index += block_index.size();
  }

  // Write all block indexes
  CHECK(MPI_File_write_at_all(file,local_block_index_start,all_block_indexes.data(),all_block_indexes.size(),MPI_BYTE,MPI_STATUS_IGNORE));
  all_block_indexes.clean_memory();
  #define all_block_indexes hide_all_block_indexes
 
  // On rank 0, write all section headers
  if (!rank) {
    Array<uint8_t> headers(header_size,false);
    size_t offset = 0;
    #define HEADER(pointer,size) \
      memcpy(headers.data()+offset,pointer,size); \
      offset += size;
    #define LE_HEADER(value) \
      value = to_little_endian(value); \
      HEADER(&value,sizeof(value));
    uint32_t version = 3;
    uint32_t section_count = sections.size();
    uint32_t section_header_size = supertensor_header_t::header_size;
    HEADER(multiple_supertensor_magic,supertensor_magic_size);
    LE_HEADER(version)
    LE_HEADER(section_count)
    LE_HEADER(section_header_size)
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
  const auto partition = empty_partition(comm_size(comm),slice);
  const auto store = new_<compacting_store_t>(0);
  const auto blocks = make_block_store(partition,rank,samples_per_section,store);
  write_sections(comm,format("%s/empty.pentago",dir),blocks,0);
}

void write_counts(const MPI_Comm comm, const string& filename, const block_store_t& blocks) {
  thread_time_t time(write_counts_kind,unevent);
  const sections_t& sections = blocks.sections;

  // Reduce win counts down to root, destroying them in the process
  const int rank = comm_rank(comm);
  const RawArray<Vector<uint64_t,3>> counts = blocks.section_counts;
  {
    const auto flat_counts = scalar_view(counts);
    if (rank) {
      CHECK(MPI_Reduce(flat_counts.data(),0,flat_counts.size(),datatype<uint64_t>(),MPI_SUM,0,comm));
      // Only the root does the actual writing
      return;
    }
    // From here on we're the root
    CHECK(MPI_Reduce(MPI_IN_PLACE,flat_counts.data(),flat_counts.size(),datatype<uint64_t>(),MPI_SUM,0,comm));
  }

  // Prepare data array
  Array<Vector<uint64_t,4>> data(counts.size(),false);
  const bool turn = sections.slice&1;
  for (int i=0;i<data.size();i++) {
    auto wins = counts[i].x, losses = counts[i].z-counts[i].y;
    if (turn)
      swap(wins,losses);
    data[i].set(sections.sections[i].sig(),wins,losses,counts[i].z);
  }

  // Pack numpy buffer.  Endianness is handled in the numpy header.
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

// The 64 bit part of big endianness is handled by numpy, so we're left with everything up to 256 bits
static inline void semiswap(Vector<super_t,2>& s) {
#if defined(BOOST_BIG_ENDIAN)
  swap(s.x.a,s.x.d);
  swap(s.x.b,s.x.c);
  swap(s.y.a,s.y.d);
  swap(s.y.b,s.y.c);
#endif
}

void write_sparse_samples(const MPI_Comm comm, const string& filename, block_store_t& blocks) {
  thread_time_t time(write_sparse_kind,unevent);
  const int rank = comm_rank(comm);
  const bool turn = blocks.sections->slice&1;

  // Mangle samples into correct output format in place
  typedef block_store_t::sample_t sample_t;
  const RawArray<sample_t> samples = blocks.samples.flat;
  if (!turn) // Black to play
    for (auto& sample : samples) {
      sample.wins.y = ~sample.wins.y;
      semiswap(sample.wins);
    }
  else // White to play
    for (auto& sample : samples) {
      sample.wins.y = ~sample.wins.y;
      swap(sample.wins.x,sample.wins.y);
      semiswap(sample.wins);
    }

  // Count total samples and send to root
  const int local_samples = samples.size();
  int total_samples;
  CHECK(MPI_Reduce((void*)&local_samples,&total_samples,1,MPI_INT,MPI_SUM,0,comm));

  // A previous version of this routine used MPI datatypes to avoid manual packing of the samples.
  // However, a 960 core run produced a double free error with the stack
  //     /lib64/libc.so.6(+0x75018)[0x2aaaaff00018]
  //     /lib64/libc.so.6(cfree+0x6c)[0x2aaaaff04fec]
  //     /opt/cray/lib64/libmpl.so.0(MPL_trfree+0x29a)[0x2aaaae65391a]
  //     /opt/cray/lib64/libmpich_gnu_47.so.1(ADIOI_Flatten+0x14d)[0x2aaaae3b238d]
  //     /opt/cray/lib64/libmpich_gnu_47.so.1(ADIOI_Flatten_datatype+0xc5)[0x2aaaae3b4135]
  //     /opt/cray/lib64/libmpich_gnu_47.so.1(ADIOI_CRAY_Exch_and_write+0xfd8)[0x2aaaae38a508]
  //     /opt/cray/lib64/libmpich_gnu_47.so.1(ADIOI_CRAY_WriteStridedColl+0x4a0)[0x2aaaae38aa50]
  //     /opt/cray/lib64/libmpich_gnu_47.so.1(MPI_File_write_ordered+0x1f1)[0x2aaaae37e001]
  //     /global/u2/g/girving/otherlab/other/install/release/lib/libpentago_core.so(_ZN7pentago3mpi20write_sparse_samplesEiRKSsRNS0_13block_store_tE+0x5a0)[0x2aaaaad7cae0]
  //     /global/u2/g/girving/otherlab/other/install/release/lib/libpentago_core.so(_ZN7pentago3mpi8toplevelEiPPc+0x2658)[0x2aaaaada8078]
  //     /lib64/libc.so.6(__libc_start_main+0xe6)[0x2aaaafea9bc6]
  //     /global/homes/g/girving/otherlab/other/install/release/bin/endgame-mpi[0x4008e9]
  // I'm not sure what the problem is, but given that it occurs in ADIOI_Flatten_datatype I'm fairly sure it'll go away if I switch to manual packing.
  Array<uint8_t> buffer;
  if (!rank) {
    // On the root, we have to write out the numpy header before our samples.
    RawArray<Vector<uint64_t,9>> all_samples(total_samples,0); // False array of all samples
    fill_numpy_header(buffer,all_samples);
  }
  // Pack samples into buffer
  int index = buffer.size();
  buffer.resize(index+(1+8)*sizeof(uint64_t)*samples.size(),false,true);
  for (const sample_t& s : samples) {
    memcpy(&buffer[index],&s.board,sizeof(s.board));
    index += sizeof(s.board);
    memcpy(&buffer[index],&s.wins,sizeof(s.wins));
    index += sizeof(s.wins);
  }

  // Write the file
  MPI_File file;
  file_open(comm,filename,&file);
  CHECK(MPI_File_write_ordered(file,buffer.data(),buffer.size(),MPI_BYTE,MPI_STATUS_IGNORE));
  CHECK(MPI_File_close(&file));
}

}
}
