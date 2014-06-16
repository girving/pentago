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
#include <geode/python/numpy.h>
#include <geode/random/Random.h>
#include <geode/structure/Hashtable.h>
#include <geode/utility/curry.h>
#include <geode/utility/ProgressIndicator.h>
#include <geode/utility/Hasher.h>
#include <geode/utility/Log.h>
#include <geode/utility/openmp.h>
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
static void filter_and_compress_and_store(Tuple<spinlock_t,ProgressIndicator>* progress, Array<uint8_t>* dst, const readable_block_store_t* blocks, local_id_t local_id, int level, bool turn) {
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

  // Adjust for the different format of readable_block_store_t and apply interleave filtering
#if PENTAGO_MPI_COMPRESS
  const auto filtered = data;
#else
  Array<Vector<super_t,2>> filtered;
#endif
  {
    thread_time_t time(filter_kind,event);
#if !PENTAGO_MPI_COMPRESS
    filtered = large_buffer<Vector<super_t,2>>(data.size(),uninit);
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

void write_sections(const MPI_Comm comm, const string& filename, const readable_block_store_t& blocks, const int level) {
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
    flat_info[info.data().flat_id] = &info.data();
#if PENTAGO_MPI_COMPRESS
    threads_schedule(CPU,curry(filter_and_compress_and_store,&progress,&compressed[info.data().flat_id],&blocks,info.key(),level,turn));
#else
    threads_schedule(CPU,curry(filter_and_compress_and_store,&progress,&compressed[info.data().flat_id],blocks.get_raw_flat(info.key),level,turn));
#endif
  }
  threads_wait_all_help();

  // Determine the base offset of each rank
  const auto sections = partition.sections->sections.raw();
  const auto& section_id = partition.sections->section_id;
  const uint64_t header_size = supertensor_magic_size+3*sizeof(uint32_t)+supertensor_header_t::header_size*sections.size();
  uint64_t local_size = pad_io;
  uint64_t previous = 0;
  {
    thread_time_t time(write_sections_kind,event);
    for (const auto& c : compressed)
      local_size += c.size();
    GEODE_ASSERT(local_size<(1u<<31));
    CHECK(MPI_Exscan(&local_size,&previous,1,datatype<uint64_t>(),MPI_SUM,comm));
  }

  // Broadcast offset of first block index to everyone
  uint64_t block_index_start = rank==ranks-1?header_size+previous+local_size:0;
  {
    thread_time_t time(write_sections_kind,event);
    CHECK(MPI_Bcast(&block_index_start,1,datatype<uint64_t>(),ranks-1,comm));
  }

  // Compute local block offsets
  Array<supertensor_blob_t> block_blobs(local_blocks+1,uninit);
  block_blobs[0].offset = header_size+previous;
  {
    thread_time_t time(write_sections_kind,event);
    for (const int b : range(local_blocks)) {
      const block_info_t& info = *flat_info[b];
      block_blobs[b].uncompressed_size = sizeof(Vector<super_t,2>)*block_shape(info.section.shape(),info.block).product();
      block_blobs[b].compressed_size = compressed[b].size();
      block_blobs[b+1].offset = block_blobs[b].offset+compressed[b].size();
    }
    block_blobs.back().offset += pad_io;
    GEODE_ASSERT(block_blobs.back().offset==header_size+previous+local_size);
  }

  // Concatenate local data into a single buffer
  {
    thread_time_t time(write_sections_kind,event);
    Array<uint8_t> buffer(int(local_size),uninit);
    int next = 0;
    for (auto& c : compressed) {
      memcpy(buffer.data()+next,c.data(),c.size());
      next += c.size();
      c.clean_memory();
    }
    if (pad_io)
      buffer[next++] = 0;
    GEODE_ASSERT(next==buffer.size());
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
    const Nested<block_blob_t> send_buffer(send_counts,uninit);
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
      Array<supertensor_blob_t,4> block_index(section_blocks,uninit);
      memset(block_index.data(),0,sizeof(supertensor_blob_t)*block_index.flat.size());
      block_indexes[sid-section_range.lo] = block_index;
      for (const int i : range(section_blocks.product())) {
        const Vector<uint8_t,4> block(decompose(section_blocks,i));
        recv_counts[partition.find_block(sections[sid],block).x]++;
      }
    }

    // Communicate
    const Nested<block_blob_t> recv_buffer(recv_counts,uninit);
    CHECK(MPI_Alltoallv(send_buffer.flat.data(),send_counts.data(),send_buffer.offsets.const_cast_().data(),block_blob_datatype,
                        recv_buffer.flat.data(),recv_counts.data(),recv_buffer.offsets.const_cast_().data(),block_blob_datatype,comm));
    CHECK(MPI_Type_free(&block_blob_datatype));

    // Packed received blobs into block indexes
    for (const auto& block_blob : recv_buffer.flat) {
      const int sid = block_blob.x;
      GEODE_ASSERT(section_range.contains(sid));
      block_indexes[sid-section_range.lo][Vector<int,4>(block_blob.y)] = block_blob.z;
    }
  }

  // Verify that all blobs are initialized
  for (const auto& block_index : block_indexes)
    for (const auto& blob : block_index.flat)
      GEODE_ASSERT(blob.uncompressed_size);

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
  uint64_t local_block_indexes_size = pad_io;
  for (const int sid : section_range)
    local_block_indexes_size += compressed_block_indexes[sid-section_range.lo].size();
  GEODE_ASSERT(local_block_indexes_size<(1u<<31));
  uint64_t previous_block_indexes_size = 0;
  CHECK(MPI_Exscan(&local_block_indexes_size,&previous_block_indexes_size,1,datatype<uint64_t>(),MPI_SUM,comm));
  const uint64_t local_block_index_start = block_index_start+previous_block_indexes_size;

  // Send all block index blobs to root
  Array<supertensor_blob_t> index_blobs(sections.size(),uninit);
  memset(index_blobs.data(),0,sizeof(supertensor_blob_t)*sections.size());
  {
    uint64_t next_block_index_offset = local_block_index_start;
    for (const int sid : section_range) {
      auto& blob = index_blobs[sid];
      blob.uncompressed_size = sizeof(supertensor_blob_t)*section_blocks(sections[sid]).product();
      blob.compressed_size = compressed_block_indexes[sid-section_range.lo].size();
      blob.offset = next_block_index_offset;
      next_block_index_offset += blob.compressed_size;
    }
  }
  CHECK(MPI_Reduce(rank?index_blobs.data():MPI_IN_PLACE,index_blobs.data(),sizeof(supertensor_blob_t)/sizeof(uint64_t)*sections.size(),datatype<uint64_t>(),MPI_SUM,0,comm));
  if (rank)
    index_blobs = Array<supertensor_blob_t>();

  // Concatenate compressed block indexes into one buffer
  Array<uint8_t> all_block_indexes(int(local_block_indexes_size),uninit);
  {
    int next_block_index = 0;
    for (const int sid : section_range) {
      const auto block_index = compressed_block_indexes[sid-section_range.lo].raw();
      memcpy(all_block_indexes.data()+next_block_index,block_index.data(),block_index.size());
      next_block_index += block_index.size();
    }
    if (pad_io)
      all_block_indexes[next_block_index++] = 0;
    GEODE_ASSERT(next_block_index==all_block_indexes.size());
  }

  // Write all block indexes
  CHECK(MPI_File_write_at_all(file,local_block_index_start,all_block_indexes.data(),all_block_indexes.size(),MPI_BYTE,MPI_STATUS_IGNORE));
  all_block_indexes.clean_memory();
  #define all_block_indexes hide_all_block_indexes

  // On rank 0, write all section headers
  if (!rank) {
    Array<uint8_t> headers(CHECK_CAST_INT(header_size),uninit);
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
      sh.pack(headers.slice(int(offset)+range(sh.header_size)));
      offset += sh.header_size;
    }
    GEODE_ASSERT(offset==header_size);
    // Write the last piece of the file
    CHECK(MPI_File_write_at(file,0,headers.data(),headers.size(),MPI_BYTE,MPI_STATUS_IGNORE));
  }

  // Done!
  CHECK(MPI_File_close(&file));
}

static Ref<const sections_t> read_section_list(const MPI_Comm comm, const vector<Ref<const supertensor_reader_t>>& tensors) {
  const int rank = comm_rank(comm);
  const auto sections = !rank ? sections_from_supertensors(tensors) : Ptr<const sections_t>();
  int count = !rank ? sections->sections.size() : 0;
  CHECK(MPI_Bcast(&count,1,datatype<int>(),0,comm));
  GEODE_ASSERT(count);
  const auto list = !rank ? sections->sections.const_cast_() : Array<section_t>(count,uninit);
  CHECK(MPI_Bcast(list.data(),CHECK_CAST_INT(memory_usage(list)),MPI_BYTE,0,comm));
  return !rank ? ref(sections) : new_<const sections_t>(list[0].sum(),list);
}

static uint64_t file_size(const MPI_Comm comm, const string& filename) {
  const int rank = comm_rank(comm);
  uint64_t size;
  if (!rank) {
    struct stat st;
    if (stat(filename.c_str(),&st))
      die("read_sections: can't stat %s: %s",filename,strerror(errno));
    size = st.st_size;
  }
  CHECK(MPI_Bcast(&size,1,datatype<uint64_t>(),0,comm));
  return size;
}

typedef Tuple<Ref<const block_partition_t>,Array<const local_block_t>,Array<const supertensor_blob_t>> ReadSectionsStart;
static ReadSectionsStart read_sections_start(const MPI_Comm comm, const string& filename,
                                             const partition_factory_t& partition_factory) {

  // Read header information and compute partitions
  const int ranks = comm_size(comm),
            rank = comm_rank(comm);
  const auto tensors = ({
    Log::Scope scope("read headers");
    !rank ? open_supertensors(filename,CPU) : vector<Ref<const supertensor_reader_t>>(); });
  const auto sections = read_section_list(comm,tensors);
  const auto partition = partition_factory(ranks,sections);
  GEODE_ASSERT(sections->total_blocks<=uint64_t(numeric_limits<int>::max()));

  // Tell all processors where their data comes from
  const auto local_blocks = partition->rank_blocks(rank);
  Array<supertensor_blob_t> blobs(local_blocks.size(),uninit);
  {
    Log::Scope scope("blob scatter");
    if (!rank) {
      Nested<supertensor_blob_t,false> all_blobs;
      all_blobs.offsets.preallocate(ranks+1);
      all_blobs.flat.preallocate(CHECK_CAST_INT(sections->total_blocks));
      for (const int r : range(ranks)) {
        all_blobs.append_empty();
        for (const auto& b : partition->rank_blocks(r))
          all_blobs.append_to_back(tensors[sections->section_id.get(b.section)]->blob(b.block));
      }
      const auto sendcounts = all_blobs.sizes();
      sendcounts *= 3;
      const auto displs = (3*all_blobs.offsets).copy();
      CHECK(MPI_Scatterv(all_blobs.flat.data(),sendcounts.data(),displs.data(),datatype<uint64_t>(),
                         blobs.data(),3*blobs.size(),datatype<uint64_t>(),0,comm));
    } else
      CHECK(MPI_Scatterv(0,0,0,datatype<uint64_t>(),
                         blobs.data(),3*blobs.size(),datatype<uint64_t>(),0,comm));
  }

  // On to the next phase
  return ReadSectionsStart(partition,local_blocks,blobs);
}

Ref<const readable_block_store_t> read_sections(const MPI_Comm comm, const string& filename, compacting_store_t& store,
                                                const partition_factory_t& partition_factory) {
  Log::Scope scope("read sections");
  const int ranks = comm_size(comm),
            rank = comm_rank(comm);

  // Read header and blob information
  const auto start = read_sections_start(comm,filename,partition_factory);
  const auto partition = start.x;
  const auto local_blocks = start.y;
  const auto blobs = start.z;

  // Allocate ordered compressed memory
  uint64_t compressed_total = 0;
  Array<int> compressed_sizes(blobs.size());
  for (const int b : range(blobs.size())) {
    compressed_total += blobs[b].compressed_size;
    compressed_sizes[b] = CHECK_CAST_INT(blobs[b].compressed_size);
  }
  GEODE_ASSERT(compressed_total<uint64_t(numeric_limits<int>::max()));
  Nested<uint8_t> compressed(compressed_sizes);
  compressed_sizes.clean_memory();

  // Slurp in the entire file
  const auto total_size = file_size(comm,filename);
  if (!rank)
    cout << "total size = "<<total_size<<endl;
  Array<char> raw;
  {
    Log::Scope scope("read data");
    const auto chunk = partition_loop(total_size,ranks,rank);
    raw.resize(CHECK_CAST_INT(chunk.size()),uninit);
    MPI_File file;
    const int r = MPI_File_open(comm,(char*)filename.c_str(),MPI_MODE_RDONLY,MPI_INFO_NULL,&file);
    if (r != MPI_SUCCESS)
      die("failed to open '%s' for reading: %s",filename,error_string(r));
    CHECK(MPI_File_read_at_all(file,chunk.lo,raw.data(),raw.size(),MPI_BYTE,MPI_STATUS_IGNORE));
    CHECK(MPI_File_close(&file));
  }

  // Just for fun, we rearrange the data using one-sided communication
  {
    Log::Scope scope("shuffle");
    MPI_Win win;
    CHECK(MPI_Win_create(raw.data(),raw.size(),1,MPI_INFO_NULL,comm,&win));
    CHECK(MPI_Win_fence(MPI_MODE_NOPUT|MPI_MODE_NOPRECEDE,win));
    for (const int b : range(blobs.size())) {
      const auto blob = blobs[b];
      const auto blob_range = blob.offset+range(blob.compressed_size);
      for (const int r : range(partition_loop_inverse(total_size,ranks,blob_range.lo),
                               partition_loop_inverse(total_size,ranks,blob_range.hi-1)+1)) {
        const auto r_range = partition_loop(total_size,ranks,r);
        GEODE_ASSERT(r_range.lo<blob_range.hi && blob_range.lo<r_range.hi);
        const auto common = range(max(blob_range.lo,r_range.lo),min(blob_range.hi,r_range.hi));
        const int size = CHECK_CAST_INT(common.size());
        CHECK(MPI_Get(&compressed(b,CHECK_CAST_INT(common.lo-blob_range.lo)),size,MPI_BYTE,
                                  r,CHECK_CAST_INT(common.lo-   r_range.lo) ,size,MPI_BYTE,win));
      }
    }
    CHECK(MPI_Win_fence(MPI_MODE_NOSUCCEED,win));
    CHECK(MPI_Win_free(&win));
    raw.clean_memory();
  }

  // Decompose data into block store
  {
    Log::Scope scope("decompress");
    const bool turn = partition->sections->slice&1;
    const auto blocks = new_<restart_block_store_t>(partition,rank,partition->rank_blocks(rank),store);
    spinlock_t progress_lock;
    ProgressIndicator progress(blobs.size());
    for (const int b : range(blobs.size()))
      threads_schedule(CPU,[=,&progress_lock,&progress](){
        const auto filtered = decompress(compressed[b],blobs[b].uncompressed_size,unevent);
        const auto unfiltered = large_buffer<Vector<super_t,2>>(filtered.size()/sizeof(Vector<super_t,2>),uninit);
        for (const int i : range(unfiltered.size())) {
          Vector<super_t,2> s;
          memcpy(&s,&filtered[sizeof(s)*i],sizeof(s));
          s = uninterleave_super(from_little_endian(s));
          unfiltered[i] = !turn ? vec(s.x,~s.y) : vec(s.y,~s.x);
        }
        blocks->set(local_blocks[b].local_id,unfiltered);
        spin_t spin(progress_lock);
        progress.progress();
      });
    threads_wait_all_help();
    blocks->store.freeze();
    return blocks;
  }
}

void read_sections_test(const MPI_Comm comm, const string& filename, const partition_factory_t& partition_factory) {
  Log::Scope scope("read sections test");
  const int ranks = comm_size(comm),
            rank = comm_rank(comm);
  if (!rank)
    cout << "ranks = "<<ranks<<endl;

  // Read header and blob information
  const auto start = read_sections_start(comm,filename,partition_factory);
  const auto partition = start.x;
  const auto local_blocks = start.y;
  const auto blobs = start.z;

  // Determine how much ordered compressed memory to allocate, but don't allocate it
  uint64_t compressed_total = 0;
  Array<int> compressed_sizes(blobs.size());
  for (const int b : range(blobs.size())) {
    compressed_total += blobs[b].compressed_size;
    compressed_sizes[b] = CHECK_CAST_INT(blobs[b].compressed_size);
  }
  if (!rank && !(compressed_total<uint64_t(numeric_limits<int>::max())))
    cout << "WARNING: compressed_total = "<<compressed_total<<", real restart job would fail"<<endl;

  // Don't slurp in the entire file
  const auto total_size = file_size(comm,filename);
  if (!rank)
    cout << "total size = "<<total_size<<endl;

  // Just for fun, we rearrange the data using one-sided communication.  Rather, we pretend to.
  {
    Log::Scope scope("shuffle");
    for (const int b : range(blobs.size())) {
      uint64_t offset = 0;
      const auto blob = blobs[b];
      const auto blob_range = blob.offset+range(blob.compressed_size);
      STATIC_ASSERT_SAME(decltype(blob_range),const Range<uint64_t>);
      typedef uint64_t UI;
      if (!(   blob_range.size()
            && UI(blob_range.lo)  <UI(total_size)
            && UI(blob_range.hi-1)<UI(total_size)))
        GEODE_ASSERT(false,format("b %d, offset %lld, cs %lld, br %lld %lld",
          b,blob.offset,blob.compressed_size,blob_range.lo,blob_range.hi));
      for (const int r : range(partition_loop_inverse(total_size,ranks,blob_range.lo),
                               partition_loop_inverse(total_size,ranks,blob_range.hi-1)+1)) {
        const auto r_range = partition_loop(total_size,ranks,r);
        GEODE_ASSERT(r_range.lo<blob_range.hi && blob_range.lo<r_range.hi);
        const auto common = range(max(blob_range.lo,r_range.lo),min(blob_range.hi,r_range.hi));
        GEODE_ASSERT(common.lo-blob_range.lo==offset);
        // Here's where we would read common.size() bytes into compressed(b,offset:?)
        offset += common.size();
      }
      GEODE_ASSERT(offset==uint64_t(compressed_sizes[b]));
    }
  }
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

void write_counts(const MPI_Comm comm, const string& filename, const accumulating_block_store_t& blocks) {
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
  Array<Vector<uint64_t,4>> data(counts.size(),uninit);
  const bool turn = sections.slice&1;
  for (int i=0;i<data.size();i++) {
    auto wins = counts[i].x, losses = counts[i].z-counts[i].y; // At this point, wins are for the player to move
    if (turn)
      swap(wins,losses); // Now wins are for black (first player), losses are for white (second player)
    data[i].set(sections.sections[i].sig(),wins,losses,counts[i].z);
  }

  // Pack numpy buffer.  Endianness is handled in the numpy header.
  auto buffer_and_size = fill_numpy_header(data);
  auto& buffer = buffer_and_size.x;
  const auto data_size = buffer_and_size.y;
  GEODE_ASSERT(data_size==sizeof(Vector<uint64_t,4>)*data.size());
  const int header_size = buffer.size();
  buffer.resize(CHECK_CAST_INT(header_size+data_size),uninit);
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

void write_sparse_samples(const MPI_Comm comm, const string& filename, accumulating_block_store_t& blocks) {
  thread_time_t time(write_sparse_kind,unevent);
  const int rank = comm_rank(comm);
  const bool turn = blocks.sections->slice&1;

  // Mangle samples into correct output format in place
  typedef accumulating_block_store_t::sample_t sample_t;
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
    buffer = fill_numpy_header(all_samples).x;
  }
  // Pack samples into buffer
  int index = buffer.size();
  buffer.resize(index+(1+8)*sizeof(uint64_t)*samples.size(),uninit);
  for (const sample_t& s : samples) {
    memcpy(&buffer[index],&s.board,sizeof(s.board));
    index += sizeof(s.board);
    memcpy(&buffer[index],&s.wins,sizeof(s.wins));
    index += sizeof(s.wins);
  }

  // Compute offsets.  MPI_File_write_ordered would do this for us, but MPI_File_write_ordered
  // is completely broken performance-wise.
  uint64_t offset = 0;
  {
    uint64_t buffer_size = buffer.size();
    CHECK(MPI_Exscan(&buffer_size,&offset,1,datatype<uint64_t>(),MPI_SUM,comm));
  }

  // Write the file
  MPI_File file;
  file_open(comm,filename,&file);
  CHECK(MPI_File_write_at_all(file,offset,buffer.data(),buffer.size(),MPI_BYTE,MPI_STATUS_IGNORE));
  CHECK(MPI_File_close(&file));
}

}
}
