// Abstract interface to files, either local or in the cloud

#include "pentago/data/file.h"
#include "pentago/utility/format.h"
#include "pentago/utility/range.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <fnmatch.h>
#include <glob.h>
#include <unistd.h>
namespace pentago {

using std::make_shared;

bool exists(const string& path) {
  return !access(path.c_str(), R_OK);
}

vector<string> listdir(const string& path) {
  DIR* dir = opendir(path.c_str());
  if (!dir)
    throw IOError(tfm::format("can't list directory '%s': %s", path, strerror(errno)));
  vector<string> paths;
  while (const dirent* d = readdir(dir))
    paths.push_back(d->d_name);
  closedir(dir);
  return paths;
}

vector<string> glob(const string& pattern) {
  glob_t pglob;
  ::glob(pattern.c_str(), GLOB_TILDE | GLOB_BRACE, nullptr, &pglob);
  vector<string> results;
  for (const auto i : range(pglob.gl_pathc))
    results.push_back(pglob.gl_pathv[i]);
  ::globfree(&pglob);
  return results;
}

read_file_t::read_file_t() {}
read_file_t::~read_file_t() {}

write_file_t::write_file_t() {}
write_file_t::~write_file_t() {}

namespace {
struct read_local_file_t : public read_file_t {
  const string path;
  const int fd;

public:
  read_local_file_t(const string& path)
    : path(path)
    , fd(open(path.c_str(), O_RDONLY, 0)) {
    if (fd < 0)
      THROW(IOError, "can't open file \"%s\" for reading: %s", path, strerror(errno));
  }

  ~read_local_file_t() {
    close(fd);
  }

  string name() const {
    return path;
  }

  string pread(RawArray<uint8_t> data, const uint64_t offset) const {
    const auto r = ::pread(fd, data.data(), data.size(), offset);
    if (r < data.size())
      return r < 0 ? strerror(errno)
                   : tfm::format("incomplete read of [%d,%d): got %d < %d",
                                 offset, offset+data.size(), r, data.size());
    return "";
  }
};

struct read_function_t : public read_file_t {
  typedef function<Array<const uint8_t>(uint64_t,int)> pread_t;

  const string name_;
  const pread_t pread_;

public:
  read_function_t(const string& name, const pread_t& pread)
    : name_(name)
    , pread_(pread) {}

  string name() const {
    return name_;
  }

  string pread(RawArray<uint8_t> data, const uint64_t offset) const {
    const auto data_ = pread_(offset,data.size());
    if (data_.size() < data.size())
      return tfm::format("incomplete read: got %d < %d", data_.size(), data.size());
    GEODE_ASSERT(data_.size()==data.size());
    data.copy(data_);
    return "";
  }
};

struct write_local_file_t : public write_file_t {
  const int fd;

public:
  write_local_file_t(const string& path)
    : fd(open(path.c_str(),O_WRONLY|O_CREAT|O_TRUNC,0644)) {
    if (fd < 0)
      THROW(IOError,"can't open file \"%s\" for writing: %s",path,strerror(errno));
  }

  ~write_local_file_t() {
    close(fd);
  }

  string pwrite(RawArray<const uint8_t> data, const uint64_t offset) {
    const auto w = ::pwrite(fd,data.data(),data.size(),offset);
    if (w<data.size())
      return w<0 ? strerror(errno) : tfm::format("incomplete write: wrote %d < %d", w, data.size());
    return "";
  }
};
}

shared_ptr<const read_file_t> read_local_file(const string& path) {
  return make_shared<read_local_file_t>(path);
}

shared_ptr<write_file_t> write_local_file(const string& path) {
  return make_shared<write_local_file_t>(path);
}

shared_ptr<const read_file_t> read_function(const string& name, const read_function_t::pread_t& pread) {
  return make_shared<read_function_t>(name,pread);
}

}
