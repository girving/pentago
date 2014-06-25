// Abstract interface to files, either local or in the cloud
#pragma once

#include <pentago/utility/debug.h>
#include <geode/array/Array.h>
#include <geode/python/Object.h>
#include <geode/utility/function.h>
namespace pentago {

// Abstract readable file
struct read_file_t : public Object {
  GEODE_DECLARE_TYPE(GEODE_EXPORT)
protected:
  read_file_t();
public:
  ~read_file_t();

  // Name for error reporting purposes only
  virtual string name() const = 0;

  // Read a block of data from a file at the given offset.  On error, return a descriptive string.
  virtual const char* pread(RawArray<uint8_t> data, const uint64_t offset) const = 0;
};

// Abstract writable file
struct write_file_t : public Object {
  GEODE_DECLARE_TYPE(GEODE_EXPORT)
protected:
  write_file_t();
public:
  ~write_file_t();

  // Write a block of data to a file at the given offset.  On error, return a descriptive string.
  virtual const char* pwrite(RawArray<const uint8_t> data, const uint64_t offset) = 0;
};

// Read access to a local file
GEODE_EXPORT Ref<const read_file_t> read_local_file(const string& path);

// Write access to a local file
GEODE_EXPORT Ref<write_file_t> write_local_file(const string& path);

// Read access via an arbitrary pread function.  Usage: data = pread(offset,size)
GEODE_EXPORT Ref<const read_file_t> read_function(const string& name,
                                                  const function<Array<const uint8_t>(uint64_t,int)>& pread);

}
