// Temporary directories
#pragma once

#include "pentago/utility/noncopyable.h"
#include <string>
namespace pentago {

using std::string;

// Make a temporary directory, then delete it upon destruction.
struct tempdir_t : public noncopyable_t {
  const string path; 

  tempdir_t(const string& name);
  ~tempdir_t();
};

}  // namespace pentago
