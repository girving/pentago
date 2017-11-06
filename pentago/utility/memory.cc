// Memory usage estimation and reporting

#include "pentago/utility/memory.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/large.h"
#include "pentago/utility/spinlock.h"
#include "pentago/utility/vector.h"
#include <iostream>
#include <stdio.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task.h>
#endif
namespace pentago {

using std::cout;
using std::endl;

static spinlock_t lock = spinlock_t();
static ssize_t total = 0;
static ssize_t peak = 0;

static const ssize_t peak_step = 214748365;
static ssize_t next_peak = peak_step;

void report_large_alloc(ssize_t change) {
  ssize_t show = 0;
  {
    spin_t spin(lock);
    total += change;
    peak = std::max(peak,total);
    if (peak >= next_peak) {
      show = peak;
      next_peak = (peak/peak_step+1)*peak_step;
    }
  }
  if (show)
    cout << format("(peak %.1fG)\n",show*10/(1<<30)*.1);
}

// Returns (total,peak)
static Vector<ssize_t,2> known() {
  spin_t spin(lock);
  return vec(total,peak);
}

#ifdef __linux__

Array<uint64_t> memory_info() {
  FILE* file = fopen("/proc/self/statm","r");
  if (!file)
    return Array<uint64_t>();
  size_t size,resident,share,text,lib,data,dt;
  int r = fscanf(file,"%zu %zu %zu %zu %zu %zu %zu",&size,&resident,&share,&text,&lib,&data,&dt);
  fclose(file);
  if (r != 7)
    return Array<uint64_t>();
  const int page = getpagesize();
  const auto known = pentago::known();
  Array<uint64_t> result(7,uninit);
  result[0] = page*size;
  result[1] = page*resident;
  result[2] = page*share;
  result[3] = page*text;
  result[4] = known[1];
  result[5] = known[0];
  result[6] = page*data;
  return result;
}

string memory_report(RawArray<const uint64_t> info) {
  if (!info.size())
    return "failed to parse /proc/self/statm";
  GEODE_ASSERT(info.size()==7);
  return format("virtual %s, resident %s, share %s, text %s, peak known %s, known %s, data %s",large(info[0]),large(info[1]),large(info[2]),large(info[3]),large(info[4]),large(info[5]),large(info[6]));
}

#elif defined(__APPLE__)

Array<uint64_t> memory_info() {
  struct mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (KERN_SUCCESS != task_info(mach_task_self(),MACH_TASK_BASIC_INFO,(task_info_t)&info,&count))
    return Array<uint64_t>();
  const auto known = pentago::known();
  Array<uint64_t> result(5,uninit);
  result[0] = info.virtual_size;
  result[1] = info.resident_size_max;
  result[2] = known[1];
  result[3] = known[0];
  result[4] = info.resident_size;
  return result;
}

string memory_report(RawArray<const uint64_t> info) {
  if (!info.size())
    return "failed";
  GEODE_ASSERT(info.size()==5);
  return format("virtual %s, peak %s, peak known %s, known %s, resident %s",large(info[0]),large(info[1]),large(info[2]),large(info[3]),large(info[4]));
}

#else

#error "Refuse to use getrusage, since it seems completely nonportable"

#endif

}
