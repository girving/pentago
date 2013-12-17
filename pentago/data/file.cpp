// Abstract interface to files, either local or in the cloud

#include <pentago/data/file.h>
#include <geode/python/Class.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
namespace pentago {

GEODE_DEFINE_TYPE(read_file_t)
GEODE_DEFINE_TYPE(write_file_t)

read_file_t::read_file_t() {}
read_file_t::~read_file_t() {}

write_file_t::write_file_t() {}
write_file_t::~write_file_t() {}

namespace {
struct read_local_file_t : public read_file_t {
  GEODE_NEW_FRIEND
  const string path;
  const int fd;

protected:
  read_local_file_t(const string& path)
    : path(path)
    , fd(open(path.c_str(),O_RDONLY,0)) {
    if (fd < 0)
      THROW(IOError,"can't open file \"%s\" for reading: %s",path,strerror(errno));
  }
public:
  ~read_local_file_t() {
    close(fd);
  }

  string name() const {
    return path;
  }

  const char* pread(RawArray<uint8_t> data, const uint64_t offset) const {
    const auto r = ::pread(fd,data.data(),data.size(),offset);
    if (r<data.size())
      return r<0 ? strerror(errno) : "incomplete read";
    return 0;
  }
};

struct read_function_t : public read_file_t {
  GEODE_NEW_FRIEND
  typedef boost::function<Array<const uint8_t>(uint64_t,int)> pread_t;

  const string name_;
  const pread_t pread_;

protected:
  read_function_t(const string& name, const pread_t& pread)
    : name_(name)
    , pread_(pread) {}
public:

  string name() const {
    return name_;
  }

  const char* pread(RawArray<uint8_t> data, const uint64_t offset) const {
    const auto data_ = pread_(offset,data.size());
    if (data_.size() < data.size())
      return "incomplete read";
    GEODE_ASSERT(data_.size()==data.size());
    data = data_;
    return 0;
  }
};

struct write_local_file_t : public write_file_t {
  GEODE_NEW_FRIEND
  const int fd;

protected:
  write_local_file_t(const string& path)
    : fd(open(path.c_str(),O_WRONLY|O_CREAT|O_TRUNC,0644)) {
    if (fd < 0)
      THROW(IOError,"can't open file \"%s\" for writing: %s",path,strerror(errno));
  }
public:
  ~write_local_file_t() {
    close(fd);
  }

  const char* pwrite(RawArray<const uint8_t> data, const uint64_t offset) {
    const auto w = ::pwrite(fd,data.data(),data.size(),offset);
    if (w<data.size())
      return w<0 ? strerror(errno) : "incomplete write";
    return 0;
  }
};
}

Ref<const read_file_t> read_local_file(const string& path) {
  return new_<read_local_file_t>(path);
}

Ref<write_file_t> write_local_file(const string& path) {
  return new_<write_local_file_t>(path);
}

Ref<const read_file_t> read_function(const string& name, const read_function_t::pread_t& pread) {
  return new_<read_function_t>(name,pread);
}

}
using namespace pentago;

void wrap_file() {
  {
    typedef read_file_t Self;
    Class<Self>("read_file_t");  
  } {
    typedef write_file_t Self;
    Class<Self>("write_file_t");  
  }

  GEODE_FUNCTION(read_local_file)
  GEODE_FUNCTION(write_local_file)
  GEODE_FUNCTION(read_function)
}
