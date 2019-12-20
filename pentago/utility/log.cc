// Logs and scopes

#include "pentago/utility/log.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/spinlock.h"
#include "pentago/utility/exceptions.h"
namespace pentago {

// Scope state
static bool active = true;
static int depth = 0;
static FILE* log_file = nullptr;

void suppress_log() {
  active = false;
}

void copy_log_to_file(const string& path) {
  if (log_file)
    fclose(log_file);
  log_file = fopen(path.c_str(), "w");
  if (!log_file)
    THROW(IOError, "Can't open '%s' for log output", path);
}

Scope::Scope(const string& name)
  : name(name), start(wall_time()) {
  if (active)
    printf("%*s%s\n", 2*depth, "", name.c_str());
  if (log_file)
    fprintf(log_file, "%*s%s\n", 2*depth, "", name.c_str());
  depth++;
}

Scope::~Scope() {
  depth--;
  const auto elapsed = wall_time() - start;
  if (active)
    printf("%*sEND %-*s%8.4f s\n", 2*depth, "", 46-2*depth, name.c_str(), elapsed.seconds());
  if (log_file)
    fprintf(log_file, "%*sEND %-*s%8.4f s\n", 2*depth, "", 46-2*depth, name.c_str(), elapsed.seconds());
}

static spinlock_t lock;

void slog(const string& msg) {
  if (active) {
    spin_t spin(lock);
    printf("%*s%s\n", 2*depth, "", msg.c_str());
  }
  if (log_file)
    fprintf(log_file, "%*s%s\n", 2*depth, "", msg.c_str());
}

}
