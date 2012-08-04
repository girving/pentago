// Memory usage estimation and reporting

#include <pentago/utility/memory.h>
#include <pentago/utility/large.h>
#include <pentago/utility/spinlock.h>
#include <other/core/math/max.h>
#include <other/core/utility/format.h>
#include <other/core/vector/Vector.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task.h>
#endif
namespace pentago {

static spinlock_t lock = spinlock_t();
static ssize_t total = 0;
static ssize_t peak = 0;

void report_large_alloc(ssize_t change) {
  spin_t spin(lock);
  total += change;
  peak = max(peak,total);
}

// Returns (total,peak)
static Vector<ssize_t,2> known() {
  spin_t spin(lock);
  return vec(total,peak);
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
  const auto known = pentago::known();
  return format("virtual %s, resident %s, share %s, text %s, peak known %s, known %s, data %s",large(page*size),large(page*resident),large(page*share),large(page*text),large(known.y),large(known.x),large(page*data));
}

#elif defined(__APPLE__)

string memory_report() {
  struct mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (KERN_SUCCESS != task_info(mach_task_self(),MACH_TASK_BASIC_INFO,(task_info_t)&info,&count))
    return "failed";
  const auto known = pentago::known();
  return format("virtual %s, peak %s, peak known %s, known %s, resident %s",large(info.virtual_size),large(info.resident_size_max),large(known.y),large(known.x),large(info.resident_size));
}

#else

#error "Refuse to use getrusage, since it seems completely nonportable"

#endif

}
