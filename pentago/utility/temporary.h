// Temporary directories
#pragma once

#include <boost/core/noncopyable.hpp>
#include <string>
namespace pentago {

using std::string;

// Make a temporary directory, then delete it upon destruction.
struct tempdir_t : public boost::noncopyable {
  const string path; 

  tempdir_t(const string& name);
  ~tempdir_t();
};

}  // namespace pentago
