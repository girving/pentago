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
#include <other/core/utility/Hasher.h>
#include <tr1/unordered_map>
#include <boost/bind.hpp>
namespace pentago {
namespace mpi {

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
 *    uncompressed.  The format is
 *
 *      char magic[20] = "pentago sections    ";
 *      uint32_t sections;
 *      supertensor_header_t headers[sections];
 *
 * 4. For now, we hard code interleave filtering (filter = 1)
 */

// Add four spaces to match size of supertensor magic string
static const char magic[21] = "pentago sections    ";

static void compress_and_store(Array<uint8_t>* dst, RawArray<const uint8_t> data, int level) {
  *dst = compress(data,level);
}

static void filter_and_compress_and_store(Array<uint8_t>* dst, RawArray<const Vector<super_t,2>> data, int level, bool turn) {
  // Adjust for the different format of block_store_t and apply interleave filtering
  Array<Vector<super_t,2>> filtered;
  {
    thread_time_t time("filter");
    filtered = aligned_buffer<Vector<super_t,2>>(data.size());
    if (!turn)
      for (int i=0;i<data.size();i++)
        filtered[i] = interleave_super(vec(data[i].x,~data[i].y));
    else
      for (int i=0;i<data.size();i++)
        filtered[i] = interleave_super(vec(~data[i].y,data[i].x));
  }
  // Compress
  *dst = compress(char_view(filtered),level);
}

void write_sections(const MPI_Comm comm, const string& filename, const block_store_t& blocks, const int level) {
  thread_time_t time("write-sections");
  const int ranks = comm_size(comm),
            rank = comm_rank(comm);
  const auto& partition = *blocks.partition;
  const int block_size = partition.block_size;
  const int filter = 1; // interleave filtering
  const bool turn = partition.slice&1;

  // Open the file
  MPI_File file;
  const int amode = MPI_MODE_WRONLY | MPI_MODE_CREATE;
  CHECK(MPI_File_open(comm,(char*)filename.c_str(),amode,MPI_INFO_NULL,&file));

  // Compress all local data
  const int local_blocks = blocks.blocks();
  vector<Array<uint8_t>> compressed(local_blocks);
  for (int b : range(blocks.blocks())) {
    const auto info = blocks.block_info[b];
    threads_schedule(CPU,boost::bind(filter_and_compress_and_store,&compressed[b],blocks.get(info.section,info.block).flat,level,turn));
  }
  threads_wait_all();

  // Determine the base offset of each rank
  const auto sections = partition.sections;
  const auto& section_id = partition.section_id;
  const uint64_t header_size = sizeof(magic)+sizeof(int)+supertensor_header_t::header_size*sections.size();
  uint64_t local_size = 0;
  for (const auto& c : compressed)
    local_size += c.size();
  OTHER_ASSERT(local_size<(1<<31));
  uint64_t previous = 0;
  CHECK(MPI_Exscan(&local_size,&previous,1,MPI_LONG_LONG_INT,MPI_SUM,comm));

  // Broadcast offset of first block index to everyone
  uint64_t block_index_start = rank==ranks-1?previous+local_size:0;
  CHECK(MPI_Bcast(&block_index_start,1,MPI_LONG_LONG_INT,ranks-1,comm));

  // Compute local block offsets
  Array<supertensor_blob_t> block_blobs(local_blocks+1,false);
  block_blobs[0].offset = header_size+previous;
  for (int b : range(local_blocks)) {
    const auto info = blocks.block_info[b];
    block_blobs[b].compressed_size = compressed[b].size();
    block_blobs[b].uncompressed_size = sizeof(Vector<super_t,2>)*block_shape(info.section.shape(),info.block,block_size).product();
    block_blobs[b+1].offset = block_blobs[b].offset+compressed[b].size();
  }
  OTHER_ASSERT(block_blobs.last().offset==header_size+previous+local_size);

  // Concatenate local data into a single buffer
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
  buffer = Array<uint8_t>(); // Deallocate

  // Now we need to write the block indexes, which requires sending all block offsets in a given section
  // to one processor for compression and writing.  We take advantage of the fact that in the current
  // partitioning scheme each rank owns blocks from at most a few sections, so it is reasonably efficient
  // to send one message per rank per section.  Each rank organizes its block offsets by section, does
  // a bunch of MPI_Isends, and then waits to receive information about any sections it owns.

  // Organize block information by section
  int remaining_owned_blocks = 0;
  Hashtable<int,Array<supertensor_blob_t,4>> block_indexes;
  typedef Tuple<Vector<int,4>,supertensor_blob_t> block_blob_t;
  BOOST_STATIC_ASSERT(sizeof(block_blob_t)==10*sizeof(int));
  Hashtable<int,Array<block_blob_t>> local_block_blobs;
  for (int b : range(local_blocks)) {
    const auto info = blocks.block_info[b];
    const int sid = section_id.get(info.section);
    if (info.block==Vector<int,4>()) {
      const auto section_blocks = pentago::mpi::section_blocks(sections[sid],block_size);
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
    int count;
    CHECK(MPI_Get_count(&status,MPI_INT,&count));
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
  Hashtable<int,Array<block_blob_t>>().swap(local_block_blobs); // Deallocate

  // Compress all block indexes
  vector<Tuple<int,Array<uint8_t>>> compressed_block_indexes;
  compressed_block_indexes.reserve(block_indexes.size());
  for (HashtableIterator<int,Array<supertensor_blob_t,4>> it(block_indexes);it.valid();it.next()) {
    compressed_block_indexes.push_back(tuple(it.key(),Array<uint8_t>()));
    threads_schedule(CPU,boost::bind(compress_and_store,&compressed_block_indexes.back().y,char_view(it.data().flat),level));
  }
  threads_wait_all();
  Hashtable<int,Array<supertensor_blob_t,4>>().swap(block_indexes); // Deallocate

  // Compute block index offsets
  uint64_t local_block_indexes_size = 0;
  for (const auto& bi : compressed_block_indexes)
    local_block_indexes_size += bi.y.size();
  OTHER_ASSERT(local_block_indexes_size<(1<<31));
  uint64_t previous_block_indexes_size = 0;
  CHECK(MPI_Exscan(&local_block_indexes_size,&previous_block_indexes_size,1,MPI_LONG_LONG_INT,MPI_SUM,comm));
  const uint64_t local_block_index_start = block_index_start+previous_block_indexes_size;

  // Send all block index blobs to root
  Array<supertensor_blob_t> index_blobs(sections.size(),false);
  memset(index_blobs.data(),0,sizeof(supertensor_blob_t)*sections.size());
  int next_block_index_offset = local_block_index_start;
  for (const auto& bi : compressed_block_indexes) {
    auto& blob = index_blobs[bi.x];
    blob.uncompressed_size = sizeof(supertensor_blob_t)*section_blocks(sections[bi.x],block_size).product();
    blob.compressed_size = bi.y.size();
    blob.offset = next_block_index_offset;
    next_block_index_offset += bi.y.size();
  }
  CHECK(MPI_Reduce(MPI_IN_PLACE,index_blobs.data(),sizeof(supertensor_blob_t)/sizeof(uint64_t)*sections.size(),MPI_LONG_LONG_INT,MPI_SUM,0,comm));
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
 
  // On rank 0, write all section headers
  if (!rank) {
    Array<uint8_t> headers(header_size,false);
    memcpy(headers.data(),magic,sizeof(magic));
    const int section_count = sections.size();
    memcpy(buffer.data()+sizeof(magic),&section_count,sizeof(int));
    for (int s=0;s<sections.size();s++) {
      supertensor_header_t sh(sections[s],block_size,filter);
      sh.valid = true;
      sh.index = index_blobs[s];
      sh.pack(headers.slice(sizeof(magic)+sizeof(int)+sh.header_size*s+range(sh.header_size)));
    }
    // Write the last piece of the file
    CHECK(MPI_File_write_at(file,0,headers.data(),headers.size(),MPI_BYTE,MPI_STATUS_IGNORE));
  }

  // Done!
  CHECK(MPI_File_close(&file));
}

void check_directory(const MPI_Comm comm, const string& dir) {
  const int slice = 24;
  const int rank = comm_rank(comm);
  const Array<const section_t> sections;
  const auto partition = new_<partition_t>(comm_size(comm),8,slice,sections);
  const auto lines = partition->rank_lines(rank,true);
  const auto blocks = new_<block_store_t>(partition,rank,lines);
  write_sections(comm,format("%s/empty.pentago"),blocks,0);
}

void write_counts(const MPI_Comm comm, const string& filename, const block_store_t& blocks) {
  thread_time_t time("write-counts");

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
  const int amode = MPI_MODE_WRONLY | MPI_MODE_CREATE;
  CHECK(MPI_File_open(MPI_COMM_SELF,(char*)filename.c_str(),amode,MPI_INFO_NULL,&file));
  CHECK(MPI_File_write(file,buffer.data(),buffer.size(),MPI_BYTE,MPI_STATUS_IGNORE));
  CHECK(MPI_File_close(&file));
}

void write_sparse_samples(const MPI_Comm comm, const string& filename, const block_store_t& blocks, const int samples_per_section) {
  thread_time_t time("write-sparse");
  const int rank = comm_rank(comm);
  const bool turn = blocks.partition->slice&1;
  const int block_size = blocks.partition->block_size;

  // Organize blocks by section
  unordered_map<section_t,unordered_map<Vector<int,4>,RawArray<const Vector<super_t,2>,4>,Hasher>,Hasher> section_blocks;
  for (int b : range(blocks.blocks()))
    section_blocks[blocks.block_info[b].section].insert(make_pair(blocks.block_info[b].block,blocks.get(b)));

  // Collect random samples
  typedef Tuple<board_t,Vector<uint64_t,8>> sample_t;
  BOOST_STATIC_ASSERT(sizeof(sample_t)==72);
  Array<sample_t> samples;
  for (const auto& s : section_blocks) {
    const auto section = s.first;
    const auto shape = section.shape();
    const auto rmin = vec(rotation_minimal_quadrants(section.counts[0]).x,
                          rotation_minimal_quadrants(section.counts[1]).x,
                          rotation_minimal_quadrants(section.counts[2]).x,
                          rotation_minimal_quadrants(section.counts[3]).x);
    const auto random = new_<Random>(hash(section));
    for (int i=0;i<samples_per_section;i++) {
      const auto index = random->uniform(Vector<int,4>(),shape),
                 block = (index+block_size-1)/block_size;
      const auto it = s.second.find(block);
      if (it != s.second.end()) {
        const auto board = quadrants(rmin[0][index[0]],
                                     rmin[1][index[1]],
                                     rmin[2][index[2]],
                                     rmin[3][index[3]]);
        auto data = it->second[index-block_size*block];
        data.y = ~data.y;
        if (turn)
          swap(data.x,data.y);
        Vector<uint64_t,8> packed;
        BOOST_STATIC_ASSERT(sizeof(packed)==sizeof(data));
        memcpy(&packed,&data,sizeof(data));
        samples.append(tuple(board,packed));
      }
    }
  }

  // Count total samples and send to root
  const int local_samples = samples.size();
  int total_samples;
  CHECK(MPI_Reduce((void*)&local_samples,&total_samples,1,MPI_LONG_LONG_INT,MPI_SUM,0,comm));

  // Open the file
  MPI_File file;
  const int amode = MPI_MODE_WRONLY | MPI_MODE_CREATE;
  CHECK(MPI_File_open(comm,(char*)filename.c_str(),amode,MPI_INFO_NULL,&file));

  // Write our data
  if (rank) {
    // On non-root processes, we only need to write out samples
    CHECK(MPI_File_write_ordered(file,samples.data(),9*samples.size(),MPI_LONG_LONG_INT,MPI_STATUS_IGNORE));
  } else {
    // On the root, we have to write out both header and our samples.  First, build the header.
    Array<uint8_t> header;
    {
      RawArray<Vector<uint64_t,9>> all_samples(total_samples,0); // False array of all samples
      fill_numpy_header(header,all_samples);
    }
    // Construct datatype combining header with local samples
    int lengths[2] = {header.size(),9*samples.size()};
    MPI_Aint displacements[2] = {(MPI_Aint)header.data(),(MPI_Aint)samples.data()};
    MPI_Datatype types[2] = {MPI_BYTE,MPI_LONG_LONG_INT};
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
}

}
}
