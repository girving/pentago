// Logs and scopes
#pragma once

#include "pentago/utility/format.h"
#include "pentago/utility/wall_time.h"
#include <boost/core/noncopyable.hpp>
namespace pentago {

using std::string;

// Silence all scopes and log messages
void suppress_log();

// Copy log messages to a file
void copy_log_to_file(const string& path);

class Scope : public boost::noncopyable {
  const string name;
  const wall_time_t start;
 public:
  Scope(const string& name);
  ~Scope();
};

// Log a message, using scope-based indentation, without formatting.  A newline is added.
void slog(const string& msg);

// Log a message, using scope-based indentation.  A newline is added.
template<class... Args> static inline void slog(const char* msg, const Args&... args) {
  slog(format(msg, args...));
}

}
