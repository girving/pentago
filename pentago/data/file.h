// Abstract interface to files, either local or in the cloud
#pragma once

#include "pentago/utility/debug.h"
#include "pentago/utility/array.h"
#include <boost/core/noncopyable.hpp>
namespace pentago {

using std::function;

// Does a file exist?
bool exists(const string& path);

// List the files in a directory
vector<string> listdir(const string& path);

// fnmatch-based glob
vector<string> glob(const string& pattern);

// Abstract readable file
struct read_file_t : private boost::noncopyable {
public:
  read_file_t();
  virtual ~read_file_t();

  // Name for error reporting purposes only
  virtual string name() const = 0;

  // Read a block of data from a file at the given offset.  On error, return a descriptive string.
  virtual string pread(RawArray<uint8_t> data, const uint64_t offset) const = 0;
};

// Abstract writable file
struct write_file_t : private boost::noncopyable {
public:
  write_file_t();
  virtual ~write_file_t();

  // Write a block of data to a file at the given offset.  On error, return a descriptive string.
  virtual string pwrite(RawArray<const uint8_t> data, const uint64_t offset) = 0;
};

// Read access to a local file
shared_ptr<const read_file_t> read_local_file(const string& path);

// Write access to a local file
shared_ptr<write_file_t> write_local_file(const string& path);

// Read access via an arbitrary pread function.  Usage: data = pread(offset,size)
shared_ptr<const read_file_t>
read_function(const string& name, const function<Array<const uint8_t>(uint64_t,int)>& pread);

}
