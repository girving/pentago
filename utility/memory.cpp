// Memory usage estimation and reporting

#include <pentago/utility/memory.h>
#include <pentago/utility/large.h>
#include <pentago/utility/spinlock.h>
#include <other/core/utility/format.h>
namespace pentago {

static spinlock_t total_lock = spinlock_t();
static ssize_t total = 0;

void report_large_alloc(ssize_t change) {
  spin_t spin(total_lock);
  total += change; 
}

static ssize_t known() {
  spin_t spin(total_lock);
  return total;
}

#ifdef __linux__

string memory_report() {
  const char* statm = "/proc/self/statm";
  FILE* file = fopen(statm,"r");
  if (!file)
    return format("failed to open %s",statm);
  size_t size,resident,share,text,lib,data,dt;
  int r = fscanf(file,"%zu %zu %zu %zu %zu %zu %zu",&size,&resident,&share,&text,&lib,&data,&dt);
  fclose(file);
  if (r != 7)
    return format("failed to parse %s",statm);
  const int page = getpagesize();
  return format("total %s, resident %s, share %s, text %s, known %s, data %s",large(page*size),large(page*resident),large(page*share),large(page*text),large(known()),large(page*data));
}

#elif defined(__APPLE__)

string memory_report() {
  struct rusage u;
  int r = getrusage(RUSAGE_SELF,&u);
  if (r)
    return strerror(errno);
  // TODO: The unit is kilobytes * ticks-of-execution.  We should correct for that somehow.
  return format("text %s, data %s, stack %s, known %s, total %s",large(u.ru_ixrss),large(u.ru_idrss),large(u.ru_isrss),large(known()), large(ru_ixrss+ru_idrss+ru_isrss));
}

#endif

}
