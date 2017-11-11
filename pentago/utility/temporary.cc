// Temporary directories

#include "pentago/utility/temporary.h"
#include "pentago/utility/const_cast.h"
#include "pentago/utility/format.h"
#include "pentago/utility/exceptions.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ftw.h>
namespace pentago {

// From https://stackoverflow.com/questions/5467725
static int unlink(const char* path, const struct stat* sb, int typeflag, struct FTW* ftwbuf) {
  return remove(path);
}
static void remove_dir(const string& path) {
  nftw(path.c_str(), unlink, 64, FTW_DEPTH | FTW_PHYS);
}

tempdir_t::tempdir_t(const string& name) {
  string path = format("/tmp/%s.XXXXXX", name); 
  if (!mkdtemp(path.data()))
    throw OSError(format("mkdtemp(%s) failed: %s", path, strerror(errno)));
  const_cast_(this->path) = path;
}

tempdir_t::~tempdir_t() {
  remove_dir(path);
}

}  // namespace pentago
