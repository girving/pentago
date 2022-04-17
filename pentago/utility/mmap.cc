#include "pentago/utility/mmap.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/memory.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
namespace pentago {

shared_ptr<void> mmap_buffer_helper(size_t size) {
  if (!size) return nullptr;
  void* start = mmap(0,size,PROT_READ|PROT_WRITE,MAP_ANON|MAP_PRIVATE,-1,0);
  if (start==MAP_FAILED) THROW(RuntimeError,"anonymous mmap failed, size = %zu",size);
  report_large_alloc(size);
  return shared_ptr<void>(start, [size](void* start) {
    munmap(start, size);
    report_large_alloc(-size);
  });
}

Array<const uint8_t> mmap_file(const string& path) {
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) THROW(IOError, strerror(errno));
  try {
    struct stat st;
    GEODE_ASSERT(!fstat(fd, &st), strerror(errno));
    const int size = CHECK_CAST_INT(uint64_t(st.st_size));
    void* start = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
    if (start == MAP_FAILED) THROW(IOError, strerror(errno));
    const shared_ptr<uint8_t> owner(
      reinterpret_cast<uint8_t*>(start),
      [fd, size](uint8_t* start) {
        munmap(start, size);
        close(fd);
      });
    return Array<const uint8_t>(vec(size), owner);
  } catch (...) {
    close(fd);
    throw;
  }
}

}
