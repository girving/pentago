// In memory and out-of-core operations on large four dimensional arrays of superscores
#pragma once

/* A supertensor / .pentago file contains win/loss information for a four dimensional slice of pentago
 * positions, defined by (1) the player attempting to win and (2) the counts of black and white
 * stones in each quadrant.  Each entry in the 4D array is itself a small 4D super_t.  We block the
 * array uniformly across all four dimensions, resulting in a 4D array of mostly uniform 4D blocks
 * (blocks at the ends of dimensions may be smaller).  Each block is independently compressed on disk.
 * The block size is chosen s.t. any two dimensional block slice fits comfortably in RAM uncompressed.
 *
 * Notes:
 * 1. The ordering of the block index and different chunks of block data is unspecified.
 * 2. Given a memory budget of M, the maximum possible block size is floor(sqrt(M/(64*420*420))).
 *    This is 9 for 1G, 13 for 2G.
 *
 * On disk, the file format is
 *
 *   // At offset 0
 *   supertensor_header_t header; // packed
 *
 *   // At header.index.offset
 *   char compressed_index_data[]; // zlib/lzma compressed data, representing
 *     supertensor_blob_t blocks[][][][]; // 4D array of information about each block
 *
 *   // Elsewhere
 *   char compressed_block_data[]; // zlib/lzma compressed block data, representing
 *     super_t filtered_block_data[][][][][2]; // filtered superscores, representing
 *       super_t block_data[][][][][2]; // sequence of (black win, white win) pairs
 *
 * All data is little endian.
 *
 * To support MPI usage, it is also possible to concatenate multiple supertensors into
 * one file using the format
 *
 *   char magic[20] = "pentago sections   \n";
 *   uint32_t version;
 *   uint32_t sections;
 *   uint32_t section_header_size;
 *   supertensor_header_t headers[sections];
 *   ... compressed per-section data ...
 *
 * All offsets are relative to the start of the entire file, and blocks from different
 * sections can occur interleaved in any order.
 *
 * File version history:
 *
 * 0 - Initial unstable version
 * 1 - Switch to storing both black and white wins
 * 2 - Change quadrant ordering to support block-wise reflection
 * 3 - Allow multiple sections in one file
 */

#include "pentago/data/file.h"
#include "pentago/base/superscore.h"
#include "pentago/base/section.h"
#include "pentago/utility/thread.h"
#include "pentago/utility/spinlock.h"
#include "pentago/utility/array.h"
#include <boost/endian/conversion.hpp>
namespace pentago {

struct supertensor_blob_t {
  uint64_t uncompressed_size; // size of the uncompressed data block
  uint64_t compressed_size; // size of the zlib/lzma compressed data block
  uint64_t offset; // offset of the start of the zlib/lzma data, or zero for undefined

  supertensor_blob_t()
    : uncompressed_size(-1)
    , compressed_size(-1)
    , offset(0) {}
};

const int supertensor_magic_size = 20;
extern const char single_supertensor_magic[21];
extern const char multiple_supertensor_magic[21];

struct supertensor_header_t {
  static constexpr int header_size = 85;

  Vector<char,20> magic; // = "pentago supertensor\n"
  uint32_t version; // see version history above
  bool valid; // was the file finished?
  uint32_t stones; // total number of stones
  section_t section; // counts of black and white stones in each quadrant
  Vector<uint16_t,4> shape; // 4D shape of the entire array
  uint32_t block_size; // dimensions of each block (except for blocks at the ends, which may be smaller)
  Vector<uint16_t,4> blocks; // shape of the block array: ceil(shape/block_size)
  uint32_t filter; // algorithm used to preprocess superscore data before compression (0 for none)
  supertensor_blob_t index; // size and location of the compressed block index

  supertensor_header_t();

  // Initialize everything except for valid and index
  supertensor_header_t(section_t section, int block_size, int filter);

  Vector<int,4> block_shape(Vector<uint8_t,4> block) const;
  void pack(RawArray<uint8_t> buffer) const;
  static supertensor_header_t unpack(RawArray<const uint8_t> buffer);
};

struct supertensor_reader_t : public boost::noncopyable {
  const shared_ptr<const read_file_t> fd;
  const supertensor_header_t header;

  // To save memory, we drop the uncompressed_size since it is computable, and store compressed_size as a uint32_t
  const Array<const uint64_t,4> offset; // absolute offset of each block in the file
  const Array<const uint32_t,4> compressed_size_; // compressed size of each block

  supertensor_reader_t(const string& path, const thread_type_t io=IO);
  supertensor_reader_t(const string& path, const shared_ptr<const read_file_t>& fd,
                       const uint64_t header_offset, const thread_type_t io=IO);
  ~supertensor_reader_t();

  // Read a block of data from disk
  Array<Vector<super_t,2>,4> read_block(Vector<uint8_t,4> block) const;

  // Read a block eventually, and call a (thread safe) function once the read completes
  typedef function<void(Vector<uint8_t,4>,Array<Vector<super_t,2>,4>)> read_cont_t;
  void schedule_read_block(Vector<uint8_t,4> block, const read_cont_t& cont) const;

  // Schedule several block reads together
  void schedule_read_blocks(RawArray<const Vector<uint8_t,4>> blocks, const read_cont_t& cont) const;

  uint64_t compressed_size(Vector<uint8_t,4> block) const;
  uint64_t uncompressed_size(Vector<uint8_t,4> block) const;
  supertensor_blob_t blob(Vector<uint8_t,4> block) const;

  // For debugging purposes
  uint64_t total_size() const;

  // File offset of the compressed index
  uint64_t index_offset() const;

  // File offsets of each block
  Array<const uint64_t,4> block_offsets() const;

 private:
  void initialize(const string& path, const uint64_t header_offset, const thread_type_t io);
};

// Locked allocation of space at the end of a file
struct next_offset_t : public boost::noncopyable {
 private:
  mutable spinlock_t lock;
  uint64_t offset;
  const Array<const uint64_t> padding;
 public:
  next_offset_t(const uint64_t start, Array<const uint64_t> padding={})
    : offset(start), padding(padding) {}

  // Reserve a block of the given size at the end of the file
  uint64_t reserve(const uint64_t size);
};

struct supertensor_writer_t : public boost::noncopyable {
  typedef supertensor_writer_t Self;

  const string path;
  shared_ptr<write_file_t> fd;
  supertensor_header_t header; // incomplete until finalize is called
  const int level; // zlib compression level
private:
  const uint64_t header_offset;
  const shared_ptr<next_offset_t> next_offset;
  const Array<supertensor_blob_t,4> index;
public:

  supertensor_writer_t(const string& path, section_t section, int block_size, int filter, int level);
  supertensor_writer_t(const string& path, const shared_ptr<write_file_t>& fd,
                       const uint64_t header_offset, const shared_ptr<next_offset_t>& next_offset,
                       section_t section, int block_size, int filter, int level);
  ~supertensor_writer_t();

  // Write a block of data to disk now, destroying it in the process.
  void write_block(Vector<uint8_t,4> block, Array<Vector<super_t,2>,4> data);

  // Write a block of data eventually, destroying it in the process.
  // The data is not necessarily actually written until finalize is called.
  void schedule_write_block(Vector<uint8_t,4> block, Array<Vector<super_t,2>,4> data);

  // Write the final index to disk and close the file
  void finalize();

  uint64_t compressed_size(Vector<uint8_t,4> block) const;
  uint64_t uncompressed_size(Vector<uint8_t,4> block) const;

private:
  void compress_and_write(supertensor_blob_t* blob, RawArray<const uint8_t> data);
  void pwrite(supertensor_blob_t* blob, Array<const uint8_t> data);
};

// Open one or more supertensors from a single file
vector<shared_ptr<const supertensor_reader_t>> open_supertensors(
    const string& path, const thread_type_t io=IO);
vector<shared_ptr<const supertensor_reader_t>> open_supertensors(
    const shared_ptr<const read_file_t>& fd, const thread_type_t io=IO);

// Write some supertensors to a single file
vector<shared_ptr<supertensor_writer_t>> supertensor_writers(
    const string& path, RawArray<const section_t> sections, const int block_size, const int filter,
    const int level, Array<const uint64_t> padding={});

// Determine the slice of a supertensor file.  Does not verify consistency.
int supertensor_slice(const string& path);

// Unfilter a filtered uncompressed block.  raw_data is destroyed.
Array<Vector<super_t,2>,4> unfilter(int filter, Vector<int,4> block_shape, Array<uint8_t> raw_data);

// Endian conversion
static inline supertensor_blob_t endian_reverse(supertensor_blob_t blob) {
  using boost::endian::endian_reverse;
  blob.uncompressed_size = endian_reverse(blob.uncompressed_size);
  blob.compressed_size = endian_reverse(blob.compressed_size);
  blob.offset = endian_reverse(blob.offset);
  return blob;
}

// Routines for writing multiple supertensors
uint64_t multiple_supertensor_header_size(const int sections);
Array<uint8_t> multiple_supertensor_header(
    RawArray<const section_t> sections, RawArray<const supertensor_blob_t> index_blobs,
    const int block_size, const int filter);

}
