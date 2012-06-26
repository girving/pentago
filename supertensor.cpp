// In memory and out-of-core operations on large four dimensional arrays of superscores

#include "supertensor.h"
#include "filter.h"
#include <other/core/array/IndirectArray.h>
#include <other/core/python/Class.h>
#include <other/core/python/to_python.h>
#include <other/core/random/Random.h>
#include <other/core/structure/HashtableIterator.h>
#include <other/core/utility/const_cast.h>
#include <other/core/utility/Log.h>
#include <other/core/utility/str.h>
#include <boost/bind.hpp>
#include <fcntl.h>
#include <unistd.h>
#include <zlib.h>
#include <lzma.h>
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

static const char* zlib_error(int z) {
  return z==Z_MEM_ERROR?"out of memory"
        :z==Z_BUF_ERROR?"insufficient output buffer space"
        :z==Z_DATA_ERROR?"incomplete or corrupted data"
        :z==Z_STREAM_ERROR?"invalid level"
        :"unknown error";
}

static const char* lzma_error(lzma_ret r) {
  switch (r) {
    case LZMA_STREAM_END:        return "end of stream was reached";
    case LZMA_NO_CHECK:          return "input stream has no integrity check";
    case LZMA_UNSUPPORTED_CHECK: return "cannot calculate the integrity check";
    case LZMA_GET_CHECK:         return "integrity check is now available";
    case LZMA_MEM_ERROR:         return "cannot allocate memory";
    case LZMA_MEMLIMIT_ERROR:    return "memory usage limit was reached";
    case LZMA_FORMAT_ERROR:      return "file format not recognized";
    case LZMA_OPTIONS_ERROR:     return "invalid or unsupported options";
    case LZMA_DATA_ERROR:        return "data is corrupt";
    case LZMA_BUF_ERROR:         return "no progress is possible";
    case LZMA_PROG_ERROR:        return "programming error";
    default:                     return "unknown error";
  }
}

static bool is_lzma(RawArray<const uint8_t> data) {
  static const uint8_t magic[6] = {0xfd,'7','z','X','Z',0};
  return data.size()>=6 && !memcmp(data.data(),magic,6);
}

static void decompress(supertensor_blob_t blob, Array<const uint8_t> compressed, const function<void(Array<uint8_t>)>& cont) {
  OTHER_ASSERT(thread_type()==CPU);

  Array<uint8_t> uncompressed;
  {
    thread_time_t time("decompress");
    size_t dest_size = blob.uncompressed_size;
    uncompressed.resize(dest_size,false,false);
    if (!is_lzma(compressed)) { // zlib
      int z = uncompress((uint8_t*)uncompressed.data(),&dest_size,compressed.data(),blob.compressed_size);
      if (z!=Z_OK)
        throw IOError(format("zlib failure in read_and_uncompress: %s",zlib_error(z)));
    } else { // lzma
      const uint32_t flags = LZMA_TELL_NO_CHECK | LZMA_TELL_UNSUPPORTED_CHECK;
      uint64_t memlimit = UINT64_MAX;
      size_t in_pos = 0, out_pos = 0;
      lzma_ret r = lzma_stream_buffer_decode(&memlimit,flags,0,compressed.data(),&in_pos,compressed.size(),uncompressed.data(),&out_pos,dest_size);
      if (r!=LZMA_OK)
        throw IOError(format("lzma failure in read_and_uncompress: %s (%d)",lzma_error(r),r));
    }
    if (dest_size != blob.uncompressed_size)
      throw IOError(format("read_and_compress: expected uncompressed size %zu, got %zu",blob.uncompressed_size,dest_size));
  }

  cont(uncompressed);
}

void read_and_uncompress(int fd, supertensor_blob_t blob, const function<void(Array<uint8_t>)>& cont) {
  // Check consistency
  OTHER_ASSERT(thread_type()==IO);
  OTHER_ASSERT(blob.compressed_size<(uint64_t)1<<31);
  OTHER_ASSERT(!blob.uncompressed_size || blob.offset);

  // Read using pread for thread safety
  Array<uint8_t> compressed;
  {
    thread_time_t time("read");
    compressed.resize(blob.compressed_size,false,false);
    ssize_t r = pread(fd,compressed.data(),blob.compressed_size,blob.offset);
    if (r<0 || r!=(ssize_t)blob.compressed_size)
      throw IOError(format("read_and_uncompress pread failed: %s",r<0?strerror(errno):"incomplete read"));
  }

  // Schedule decompression
  schedule(CPU,boost::bind(decompress,blob,compressed,cont));
}

void supertensor_writer_t::pwrite(supertensor_blob_t* blob, Array<const uint8_t> data) {
  OTHER_ASSERT(thread_type()==IO);

  // Choose offset
  {
    lock_t lock(offset_mutex);
    blob->offset = next_offset;
    next_offset += data.size();
  }

  // Write using pwrite for thread safety
  thread_time_t time("write");
  ssize_t w = ::pwrite(fd.fd,data.data(),data.size(),blob->offset);
  if (w < 0 || w < (ssize_t)data.size())
    throw IOError(format("failed to write compressed block to supertensor file: %s",w<0?strerror(errno):"incomplete write"));
}

Array<uint8_t> compress(RawArray<const uint8_t> data, int level) {
  if (level<20) { // zlib
    size_t dest_size = compressBound(data.size());
    OTHER_ASSERT(dest_size<(uint64_t)1<<31);
    Array<uint8_t> compressed(dest_size,false);
    int z = compress2(compressed.data(),&dest_size,(uint8_t*)data.data(),data.size(),level);
    if (z!=Z_OK)
      throw IOError(format("zlib failure in compress_and_write: %s",zlib_error(z)));
    return compressed.slice_own(0,dest_size);
  } else { // lzma
    size_t dest_size = lzma_stream_buffer_bound(data.size());
    OTHER_ASSERT(dest_size<(uint64_t)1<<31);
    Array<uint8_t> compressed(dest_size,false);
    size_t pos = 0;
    lzma_ret r = lzma_easy_buffer_encode(level-20,LZMA_CHECK_CRC64,0,data.data(),data.size(),compressed.data(),&pos,dest_size);
    if (r!=LZMA_OK)
      throw RuntimeError(format("lzma compression error: %s (%d)",lzma_error(r),r));
    return compressed.slice_own(0,pos);
  }
}

void supertensor_writer_t::compress_and_write(supertensor_blob_t* blob, RawArray<const uint8_t> data) {
  OTHER_ASSERT(thread_type()==CPU);

  // Compress
  thread_time_t time("compress");
  blob->uncompressed_size = data.size();
  Array<uint8_t> compressed = compress(data,level);
  blob->compressed_size = compressed.size();

  // Schedule write
  schedule(IO,boost::bind(&Self::pwrite,this,blob,compressed));
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

static void save_index(Vector<int,4> blocks, Array<const supertensor_blob_t,4>* dst, Array<const uint8_t> src) {
  OTHER_ASSERT(src.size()==(int)sizeof(supertensor_blob_t)*blocks.product());
  *dst = Array<const supertensor_blob_t,4>(blocks,(const supertensor_blob_t*)src.data(),src.borrow_owner());
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
  ssize_t r = pread(fd.fd,buffer,header_size,0);
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
  if (h.version != 2)
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
  OTHER_ASSERT((h.block_size&1)==0);
  OTHER_ASSERT(h.blocks==(h.shape+h.block_size-1)/h.block_size);

  // Read block index
  Array<const supertensor_blob_t,4> index;
  schedule(IO,boost::bind(read_and_uncompress,fd.fd,h.index,function<void(Array<uint8_t>)>(boost::bind(save_index,Vector<int,4>(h.blocks),&index,_1))));
  wait_all();
  OTHER_ASSERT((index.shape==Vector<int,4>(h.blocks)));
  const_cast_(this->index) = index;
}

supertensor_reader_t::~supertensor_reader_t() {}

static void save(Array<Vector<super_t,2>,4>* dst, Vector<int,4> block, Array<Vector<super_t,2>,4> src) {
  *dst = src;
}

Array<Vector<super_t,2>,4> supertensor_reader_t::read_block(Vector<int,4> block) const {
  Array<Vector<super_t,2>,4> data;
  schedule_read_block(block,boost::bind(save,&data,_1,_2));
  wait_all();
  OTHER_ASSERT(data.shape==header.block_shape(block));
  return data;
}

static void unfilter(int filter, Vector<int,4> block_shape, Array<uint8_t> raw_data, const function<void(Array<Vector<super_t,2>,4>)>& cont) {
  OTHER_ASSERT(thread_type()==CPU);
  OTHER_ASSERT(raw_data.size()==(int)sizeof(Vector<super_t,2>)*block_shape.product());
  Array<Vector<super_t,2>,4> data(block_shape,(Vector<super_t,2>*)raw_data.data(),raw_data.borrow_owner());
  switch (filter) {
    case 0: break;
    case 1: uninterleave(data.flat); break;
    default: throw ValueError(format("supertensor_reader_t::read_block: unknown filter %d",filter));
  }
  cont(data);
}

void supertensor_reader_t::schedule_read_block(Vector<int,4> block, const function<void(Vector<int,4>,Array<Vector<super_t,2>,4>)>& cont) const {
  schedule_read_blocks(RawArray<const Vector<int,4>>(1,&block),cont);
}

void supertensor_reader_t::schedule_read_blocks(RawArray<const Vector<int,4>> blocks, const function<void(Vector<int,4>,Array<Vector<super_t,2>,4>)>& cont) const {
  vector<function<void()>> jobs;
  for (auto block : blocks) {
    OTHER_ASSERT(index.valid(block));
    jobs.push_back(boost::bind(read_and_uncompress,fd.fd,index[block],
                                 function<void(Array<uint8_t>)>(boost::bind(unfilter,header.filter,header.block_shape(block),_1,
                                   function<void(Array<Vector<super_t,2>,4>)>(boost::bind(cont,block,_1))))));
  }
  schedule(IO,jobs);
}

// Write header at offset 0, and return the header size
static uint64_t write_header(int fd, const supertensor_header_t& h) {
  const auto fields = header_fields();
  const int header_size = total_size(fields);
  char buffer[header_size];
  for (auto f : fields)
    memcpy(buffer+f.file_offset,(char*)&h+f.header_offset,f.size);
  ssize_t w = pwrite(fd,buffer,header_size,0);
  if (w < 0 || w < header_size)
    throw IOError(format("failed to write header to supertensor file: %s",w<0?strerror(errno):"incomplete write"));
  return header_size;
}

supertensor_writer_t::supertensor_writer_t(const string& path, section_t section, int block_size, int filter, int level)
  : path(check_extension(path))
  , fd(path,O_WRONLY|O_CREAT|O_TRUNC,0644)
  , header()
  , level(level) {
  if (fd.fd < 0)
    throw IOError(format("can't open supertensor file \"%s\" for writing: %s",path,strerror(errno)));
  if (block_size&1)
    throw ValueError(format("supertensor block size must be even (not %d) to support block-wise reflection",block_size));

  // Initialize header
  supertensor_header_t h;
  memcpy(h.magic,magic,20);
  h.version = 2;
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
  next_offset = write_header(fd.fd,h);

  // Initialize block index to all undefined
  const_cast_(index) = Array<supertensor_blob_t,4>(Vector<int,4>(h.blocks));
}

supertensor_writer_t::~supertensor_writer_t() {
  if (!header.valid) {
    // File wasn't finished (due to an error or a failure to call finalize), so delete it
    int r = unlink(path.c_str());
    if (r < 0 && errno != ENOENT)
      cerr << format("failed to remove incomplete supertensor file \"%s\": %s",path,strerror(errno)) << endl;
  }
}

static void filter(int filter, Array<Vector<super_t,2>,4> data, const function<void(Array<uint8_t>)>& cont) {
  OTHER_ASSERT(thread_type()==CPU);
  switch (filter) {
    case 0: break;
    case 1: interleave(data.flat); break;
    default: throw ValueError(format("supertensor_writer_t::write_block: unknown filter %d",filter));
  }
  schedule(CPU,boost::bind(cont,char_view_own(data.flat))); 
}

void supertensor_writer_t::write_block(Vector<int,4> block, Array<Vector<super_t,2>,4> data) {
  schedule_write_block(block,data);
  wait_all();
}

void supertensor_writer_t::schedule_write_block(Vector<int,4> block, Array<Vector<super_t,2>,4> data) {
  OTHER_ASSERT(data.shape==header.block_shape(block));
  OTHER_ASSERT(!index[block].offset); // Don't write the same block twice
  schedule(CPU,boost::bind(filter,header.filter,data,function<void(Array<uint8_t>)>(boost::bind(&Self::compress_and_write,this,&index[block],_1))));
}

void supertensor_writer_t::finalize() {
  if (fd.fd < 0 || header.valid)
    return;

  // Check if all blocks have been written
  wait_all();
  for (auto blob : index.flat)
    if (!blob.offset)
      throw RuntimeError(format("can't finalize incomplete supertensor file \"%s\"",path));

  // Write index
  supertensor_header_t h = header;
  schedule(CPU,boost::bind(&Self::compress_and_write,this,&h.index,char_view_own(index.flat)));
  wait_all();

  // Finalize header
  h.valid = true;
  write_header(fd.fd,h);
  header = h;
  fd.close();
}

uint64_t supertensor_reader_t::compressed_size(Vector<int,4> block) const {
  header.block_shape(block); // check validity
  return index[block].compressed_size;
}

uint64_t supertensor_reader_t::uncompressed_size(Vector<int,4> block) const {
  header.block_shape(block); // check validity
  return index[block].uncompressed_size;
}

uint64_t supertensor_writer_t::compressed_size(Vector<int,4> block) const {
  header.block_shape(block); // check validity
  return index[block].compressed_size;
}

uint64_t supertensor_writer_t::uncompressed_size(Vector<int,4> block) const {
  header.block_shape(block); // check validity
  return index[block].uncompressed_size;
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
    .OTHER_METHOD(compressed_size)
    .OTHER_METHOD(uncompressed_size)
    ;}

  {typedef supertensor_writer_t Self;
  Class<Self>("supertensor_writer_t")
    .OTHER_INIT(const string&,section_t,int,int,int)
    .OTHER_CONST_FIELD(header)
    .OTHER_METHOD(write_block)
    .OTHER_METHOD(finalize)
    .OTHER_METHOD(compressed_size)
    .OTHER_METHOD(uncompressed_size)
    ;}

  OTHER_FUNCTION_2(compress,pentago::compress)
}
