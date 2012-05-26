// In memory and out-of-core operations on large four dimensional arrays of superscores

#include "supertensor.h"
#include <other/core/array/IndirectArray.h>
#include <other/core/python/Class.h>
#include <other/core/python/to_python.h>
#include <other/core/random/Random.h>
#include <other/core/structure/HashtableIterator.h>
#include <other/core/utility/const_cast.h>
#include <other/core/utility/Log.h>
#include <fcntl.h>
#include <unistd.h>
#include <zlib.h>
namespace pentago {

using std::cout;
using std::cerr;
using std::endl;

struct supertensor_header_py : public Object, public supertensor_header_t {
  OTHER_DECLARE_TYPE
protected:
  supertensor_header_py(supertensor_header_t h)
    : supertensor_header_t(h) {}
};

OTHER_DEFINE_TYPE(supertensor_header_py)
OTHER_DEFINE_TYPE(supertensor_reader_t)
OTHER_DEFINE_TYPE(supertensor_writer_t)

static PyObject* to_python(const supertensor_header_t& h) {
  return to_python(new_<supertensor_header_py>(h));
}

Vector<int,4> supertensor_header_t::block_shape(Vector<int,4> block) const {
  OTHER_ASSERT(block.min()>=0 && (Vector<int,4>(blocks)-block).min()>=0);
  Vector<int,4> bs;
  for (int i=0;i<4;i++)
    bs[i] = block[i]+1<blocks[i]?block_size:shape[i]-block_size*(blocks[i]-1);
  return bs;
}

const char* zlib_error(int z) {
  return z==Z_MEM_ERROR?"out of memory"
        :z==Z_BUF_ERROR?"insufficient output buffer space"
        :z==Z_DATA_ERROR?"incomplete or corrupted data"
        :z==Z_STREAM_ERROR?"invalid level"
        :"unknown error";
}

static void read_and_uncompress(int fd, RawArray<uint8_t> data, supertensor_blob_t blob) {
  // Check consistency
  OTHER_ASSERT(blob.compressed_size<(uint64_t)1<<31);
  OTHER_ASSERT(blob.uncompressed_size==(uint64_t)data.size());
  OTHER_ASSERT(!blob.uncompressed_size || blob.offset);
  if (!blob.uncompressed_size)
    return;

  // Seek
  off_t o = lseek(fd,blob.offset,SEEK_SET);
  if (o < 0)
    throw IOError(format("read_and_uncompress: lseek failed, %s",strerror(errno)));
  OTHER_ASSERT(o==(off_t)blob.offset);

  // Read
  Array<uint8_t> compressed(blob.compressed_size,false);
  OTHER_ASSERT(read(fd,compressed.data(),blob.compressed_size)==(ssize_t)blob.compressed_size);

  // Decompress
  size_t dest_size = blob.uncompressed_size;
  int z = uncompress((uint8_t*)data.data(),&dest_size,compressed.data(),blob.compressed_size);
  if (z!=Z_OK)
    throw IOError(format("zlib failure in read_and_uncompress: %s",zlib_error(z)));
  if (dest_size != blob.uncompressed_size)
    throw IOError(format("read_and_compress: expected uncompressed size %zu, got %zu",blob.uncompressed_size,dest_size));
}

static supertensor_blob_t compress_and_write(int fd, RawArray<const uint8_t> data, int level, bool verbose) {
  // Initialize blob
  supertensor_blob_t blob;
  blob.uncompressed_size = data.size();

  // Remember offset
  off_t offset = lseek(fd,0,SEEK_CUR);
  if (offset < 0)
    throw IOError(format("compress_and_write: lseek failed, %s",strerror(errno)));
  if (!offset)
    throw IOError("compress_and_write: writing a compressed block at offset zero is disallowed");
  blob.offset = offset;

  // Compress
  if (verbose)
    Log::time("compress");
  size_t dest_size = compressBound(blob.uncompressed_size);
  Array<uint8_t> compressed(dest_size,false);
  int z = compress2(compressed.data(),&dest_size,(uint8_t*)data.data(),blob.uncompressed_size,level);
  if (z!=Z_OK)
    throw IOError(format("zlib failure in compress_and_write: %s",zlib_error(z)));
  OTHER_ASSERT(dest_size<(uint64_t)1<<31);
  blob.compressed_size = dest_size;

  // Write
  if (verbose)
    Log::time("write");
  ssize_t w = write(fd,compressed.data(),dest_size);
  if (verbose)
    Log::stop_time();
  if (w < 0 || w < (ssize_t)dest_size)
    throw IOError(format("failed to write compressed block to supertensor file: %s",w<0?strerror(errno):"incomplete write"));
  return blob;
}

template<class TA> static RawArray<typename CopyConst<uint8_t,typename TA::Element>::type> char_view(const TA& data) {
  uint64_t size = sizeof(typename TA::Element)*(size_t)data.size();
  OTHER_ASSERT(size<(uint64_t)1<<31);
  typedef typename CopyConst<uint8_t,typename TA::Element>::type C;
  return RawArray<C>(size,(C*)data.data());
}

static const char magic[21] = "pentago supertensor\n";

fildes_t::fildes_t(const string& path, int flags, mode_t mode)
  : fd(open(path.c_str(),flags,mode)) {}

fildes_t::~fildes_t() {
  close();
}

void fildes_t::close() {
  if (fd >= 0) {
    ::close(fd);
    fd = -1;
  }
}

static const string& check_extension(const string& path) {
  OTHER_ASSERT(path.size()>=8);
  OTHER_ASSERT(path.substr(path.size()-8)==".pentago");
  return path;
}

struct field_t {
  int header_offset, file_offset, size;

  field_t(int header_offset, int file_offset, int size)
    : header_offset(header_offset), file_offset(file_offset), size(size) {}
};

// List of offset,size pairs
static Array<const field_t> header_fields() {
  Array<field_t> fields;
  supertensor_header_t h;
  int offset = 0;
  #define FIELD(f) \
    fields.append(field_t((char*)&h.f-(char*)&h,offset,sizeof(h.f))); \
    offset += sizeof(h.f);
  FIELD(magic);
  FIELD(version);
  FIELD(valid);
  FIELD(stones);
  FIELD(section);
  FIELD(shape);
  FIELD(block_size);
  FIELD(blocks);
  FIELD(filter);
  FIELD(index);
  #undef FIELD
  return fields;
}

static int total_size(RawArray<const field_t> fields) {
  return fields.project<int,&field_t::size>().sum();
}

supertensor_reader_t::supertensor_reader_t(const string& path)
  : fd(check_extension(path),O_RDONLY)
  , header() {
  if (fd.fd < 0)
    throw IOError(format("can't open supertensor file \"%s\" for reading: %s",path,strerror(errno)));

  // Read header
  const auto fields = header_fields();
  const int header_size = total_size(fields);
  char buffer[header_size];
  ssize_t r = read(fd.fd,buffer,header_size);
  if (r < 0)
    throw IOError(format("invalid supertensor file \"%s\": error reading header, %s",path,strerror(errno)));
  if (r < header_size)
    throw IOError(format("invalid supertensor file \"%s\": unexpected end of file during header",path));
  supertensor_header_t h;
  for (auto f : fields)
    memcpy((char*)&h+f.header_offset,buffer+f.file_offset,f.size);
  const_cast_(header) = h;

  // Verify header
  if (memcmp(h.magic,magic,20))
    throw IOError(format("invalid supertensor file \"%s\": incorrect magic string",path));
  if (h.version != 1)
    throw IOError(format("supertensor file \"%s\" has unknown version %d",path,h.version));
  if (!h.valid)
    throw IOError(format("supertensor file \"%s\" is marked invalid",path));
  OTHER_ASSERT(h.stones==h.section.counts.sum().sum());
  for (int i=0;i<4;i++) {
    OTHER_ASSERT(h.section.counts[i].max()<=9);
    OTHER_ASSERT(h.section.counts[i].sum()<=9);
  }
  OTHER_ASSERT((Vector<int,4>(h.shape)==h.section.shape()));
  OTHER_ASSERT(1<=h.block_size && h.block_size<=27);
  OTHER_ASSERT(h.blocks==(h.shape+h.block_size-1)/h.block_size);

  // Read block index
  Array<int> index_shape;
  index_shape.copy(Vector<int,4>(h.blocks));
  NdArray<supertensor_blob_t> index(index_shape,false);
  read_and_uncompress(fd.fd,char_view(index.flat),h.index);
  const_cast_(this->index) = index;
}

supertensor_reader_t::~supertensor_reader_t() {}

void supertensor_reader_t::read_block(Vector<int,4> block, NdArray<Vector<super_t,2>> data) const {
  OTHER_ASSERT(data.rank()==4);
  OTHER_ASSERT((Vector<int,4>(data.shape.subset(vec(0,1,2,3)))==header.block_shape(block)));
  read_and_uncompress(fd.fd,char_view(data.flat),index[block]);
  if (header.filter)
    OTHER_ASSERT(false);
}

static void write_header(int fd, const supertensor_header_t& h) {
  const auto fields = header_fields();
  const int header_size = total_size(fields);
  char buffer[header_size];
  for (auto f : fields)
    memcpy(buffer+f.file_offset,(char*)&h+f.header_offset,f.size);
  ssize_t w = write(fd,buffer,header_size);
  if (w < 0 || w < header_size)
    throw IOError(format("failed to write header to supertensor file: %s",w<0?strerror(errno):"incomplete write"));
}

supertensor_writer_t::supertensor_writer_t(const string& path, section_t section, int block_size, int filter, int level)
  : path(check_extension(path))
  , fd(path,O_WRONLY|O_CREAT|O_TRUNC,0644)
  , header()
  , level(level) {
  if (fd.fd < 0)
    throw IOError(format("can't open supertensor file \"%s\" for writing: %s",path,strerror(errno)));

  // Initialize header
  supertensor_header_t h;
  memcpy(h.magic,magic,20);
  h.version = 1;
  h.stones = section.counts.sum().sum();
  h.section = section;
  h.shape = Vector<uint16_t,4>(section.shape());
  h.block_size = block_size;
  h.blocks = (h.shape+block_size-1)/block_size;
  h.filter = filter;
  // valid and index will be finalized during the destructor
  h.valid = false;
  const_cast_(header) = h;

  // Write preliminary header
  write_header(fd.fd,h);

  // Initialize block index to all undefined
  Array<int> index_shape;
  index_shape.copy(Vector<int,4>(h.blocks));
  const_cast_(index) = NdArray<supertensor_blob_t>(index_shape);
}

supertensor_writer_t::~supertensor_writer_t() {
  if (!header.valid) {
    // File wasn't finished (due to an error or a failure to call finalize), so delete it
    int r = unlink(path.c_str());
    if (r < 0 && errno != ENOENT)
      cerr << format("failed to remove incomplete supertensor file \"%s\": %s",path,strerror(errno)) << endl;
  }
}

void supertensor_writer_t::write_block(Vector<int,4> block, NdArray<const Vector<super_t,2>> data) {
  OTHER_ASSERT(data.rank()==4);
  OTHER_ASSERT((Vector<int,4>(data.shape.subset(vec(0,1,2,3)))==header.block_shape(block)));
  OTHER_ASSERT(!index[block].offset); // Don't write the same block twice
  if (header.filter)
    OTHER_ASSERT(false);
  index[block] = compress_and_write(fd.fd,char_view(data.flat),level,true);
}

void supertensor_writer_t::finalize() {
  if (fd.fd < 0 || header.valid)
    return;

  // Check if all blocks have been written
  for (auto blob : index.flat)
    if (!blob.offset)
      throw RuntimeError(format("can't finalize incomplete supertensor file \"%s\"",path));

  // Write index
  supertensor_header_t h = header;
  h.index = compress_and_write(fd.fd,char_view(index.flat),level,false);

  // Finalize header
  h.valid = true;
  off_t o = lseek(fd.fd,0,SEEK_SET);
  if (o < 0)
    throw IOError(format("read_and_uncompress: lseek failed, %s",strerror(errno)));
  write_header(fd.fd,h);
  header = h;
  fd.close();
}

uint64_t supertensor_writer_t::compressed_size(Vector<int,4> block) const {
  header.block_shape(block); // check validity
  return index[block].compressed_size;
}

}
using namespace pentago;

void wrap_supertensor() {
  {typedef supertensor_header_py Self;
  Class<Self>("supertensor_header_t")
    .OTHER_FIELD(version)
    .OTHER_FIELD(valid)
    .OTHER_FIELD(stones)
    .OTHER_FIELD(section)
    .OTHER_FIELD(shape)
    .OTHER_FIELD(block_size)
    .OTHER_FIELD(blocks)
    .OTHER_FIELD(filter)
    .OTHER_METHOD(block_shape)
    ;}

  {typedef supertensor_reader_t Self;
  Class<Self>("supertensor_reader_t")
    .OTHER_INIT(const string&)
    .OTHER_FIELD(header)
    .OTHER_METHOD(read_block)
    ;}

  {typedef supertensor_writer_t Self;
  Class<Self>("supertensor_writer_t")
    .OTHER_INIT(const string&,section_t,int,int,int)
    .OTHER_CONST_FIELD(header)
    .OTHER_METHOD(write_block)
    .OTHER_METHOD(finalize)
    .OTHER_METHOD(compressed_size)
    ;}
}
