// Thread utilities

#include "pentago/utility/thread.h"
#include "pentago/utility/box.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/range.h"
#include "pentago/utility/spinlock.h"
#include "pentago/utility/wall_time.h"
#include "pentago/utility/log.h"
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <mutex>
#include <deque>
#include <set>
namespace pentago {

/****************** Configuration *****************/

// Flip to enable history tracking
#define HISTORY 0

// Whether to use blocking pthread locking or nonblocking spinlocks.
// Use nonblocking mode (0) except for testing purposes.
#define BLOCKING 0

// Set to nonzero to enable PAPI support.  If on, the set of counters can be
// controlled with the PENTAGO_PAPI environment variable.
#define PAPI_MAX 0

/********************** Setup *********************/

using std::deque;
using std::set;
using std::exception;

#define CHECK(exp) ({ \
  int r_ = (exp); \
  if (r_) \
    THROW(RuntimeError,"thread_pool_t: %s failed, %s",#exp,strerror(r_)); \
  })

/****************** PAPI support ******************/

static_assert(PAPI_MAX >= 0,"");

#if PAPI_MAX
#ifndef PENTAGO_PAPI
#error "PAPI support not available"
#endif
}
#include <papi.h>
namespace pentago {
#endif

static bool papi_initialized = false;
static int papi_count = 0;
static int papi_events[PAPI_MAX];

static string papi_event_name(const int event) {
#if !PAPI_MAX
  die("Compiled without PAPI support");
#else
  char name[PAPI_MAX_STR_LEN];
  if (const int r = PAPI_event_code_to_name(event,name))
    die("Failed to convert PAPI event %d to string: %s",event,PAPI_strerror(r));
  return name;
#endif
}

static void init_papi() {
  if (papi_initialized)
    return;
  papi_initialized = true;
#if PAPI_MAX
  const int version = PAPI_library_init(PAPI_VER_CURRENT);
  if (version != PAPI_VER_CURRENT)
    die("PAPI_library_init failed: %s",
        version>0 ? format("version mismatch: expected %d, got %d",PAPI_VER_CURRENT,version)
                  : PAPI_strerror(version));
  if (const int r = PAPI_thread_init(pthread_self))
    die("PAPI_thread_init failed: %s",PAPI_strerror(r));
#endif
  const char* events = getenv("PENTAGO_PAPI");
  if (!events)
    return;
#if !PAPI_MAX
  die("PAPI events '%s' requested, but PAPI support was disabled at compile time",events);
#else
  const char* white = " \t\v\f\r\n";
  string copy = events;
  char *p = copy.data(), *save;
  while (char* event = strtok_r(p,white,&save)) {
    if (papi_count == PAPI_MAX)
      die("Too many PAPI events requested, limit is %d, got %s",PAPI_MAX,events);
    if (const int r = PAPI_event_name_to_code(event,&papi_events[papi_count]))
      die("Can't parse PAPI event '%s': %s",event,PAPI_strerror(r));
    papi_count++;
    p = 0; // Prepare for next strtok_r call
  }
#endif
}

bool papi_enabled() {
  return PAPI_MAX > 0;
}

vector<string> papi_event_names() {
  GEODE_ASSERT(papi_initialized);
  vector<string> names;
  for (const int p : range(papi_count))
    names.push_back(papi_event_name(papi_events[p]));
  return names;
}

#if PAPI_MAX
static int create_papi_eventset() {
  GEODE_ASSERT(papi_initialized);
  int set = PAPI_NULL;
  if (!papi_count)
    return set;
  if (const int r = PAPI_create_eventset(&set))
    die("Failed to create PAPI eventset: %s",PAPI_strerror(r));
  for (const int i : range(papi_count)) {
    const auto event = papi_events[i];
    if (const int r = PAPI_add_event(set,event))
      die("Failed to add event '%s' (0x%x) to eventset: %s",papi_event_name(event),event,PAPI_strerror(r));
  }
  return set;
}
#endif

/****************** thread_time_t *****************/

struct time_table_t;

struct time_entry_t {
  wall_time_t total, local, start;
  event_t event;
  GEODE_DEBUG_ONLY(time_table_t* thread;)
#if HISTORY
  Array<history_t> history;
#endif
#if PAPI_MAX
  int papi_eventset;
  papi_t papi_total[PAPI_MAX], papi_local[PAPI_MAX];
#endif

  time_entry_t()
    : total(0), local(0), start(0), event(0) {
#if PAPI_MAX
    papi_eventset = PAPI_NULL;
    memset(papi_total,0,sizeof(papi_total));
    memset(papi_local,0,sizeof(papi_local));
#endif
  }
};

struct time_table_t {
  thread_type_t type;
  time_entry_t times[_time_kinds];
  GEODE_DEBUG_ONLY(time_kind_t active_kind;)

  time_table_t(thread_type_t type)
    : type(type) {
    GEODE_DEBUG_ONLY(
      for (auto& entry : times)
        entry.thread = this;
      active_kind = (time_kind_t)-1;
    )
#if PAPI_MAX
    // Create one eventset for the entire thread to avoid hitting the file descriptor limit
    // For details, see https://lists.eecs.utk.edu/pipermail/ptools-perfapi/2014-April/003205.html.
    const auto eventset = create_papi_eventset();
    for (auto& time : times)
      time.papi_eventset = eventset;
#endif
  }
};
vector<time_table_t*> time_tables;
static spinlock_t time_lock;

struct time_info_t {
  pthread_key_t key;
  wall_time_t local_start;

  time_info_t()
    : local_start(0) {
    pthread_key_create(&key,0);
  }

  void init_thread(thread_type_t type) {
    if (!pthread_getspecific(key)) {
      spin_t spin(time_lock);
      time_tables.push_back(new time_table_t(type));
      pthread_setspecific(key,time_tables.back());
    }
  }
};
static time_info_t time_info;

thread_type_t thread_type() {
  auto table = (time_table_t*)pthread_getspecific(time_info.key);
  GEODE_ASSERT(table);
  return table->type;
}

#if PENTAGO_TIMING

static inline time_entry_t& time_entry(time_kind_t kind) {
  auto table = (time_table_t*)pthread_getspecific(time_info.key);
  GEODE_ASSERT(table);
  return table->times[kind];
}

thread_time_t::thread_time_t(time_kind_t kind, event_t event)
  : entry(&time_entry(kind)) {
  entry->start = wall_time();
  entry->event = event;
  GEODE_DEBUG_ONLY(
    auto& active = entry->thread->active_kind;
    if (active != (time_kind_t)-1)
      die("Tried to start time kind %d inside time kind %d",kind,active);
    active = kind;
  )
#if PAPI_MAX
  if (entry->papi_eventset != PAPI_NULL)
    if (const int r = PAPI_start(entry->papi_eventset))
      die("PAPI_start failed: eventset %d, %s",entry->papi_eventset,PAPI_strerror(r));
#endif
}

thread_time_t::~thread_time_t() {
  stop();
}

void thread_time_t::stop() {
  if (entry) {
    GEODE_DEBUG_ONLY(entry->thread->active_kind = (time_kind_t)-1;)
#if PAPI_MAX
    if (entry->papi_eventset != PAPI_NULL) {
      papi_t values[papi_count];
      if (const int r = PAPI_stop(entry->papi_eventset,values))
        die("PAPI_stop failed: %s",PAPI_strerror(r));
      for (int p=0;p<papi_count;p++)
        entry->papi_local[p] += values[p];
    }
#endif
    wall_time_t now = wall_time();
#if HISTORY
    entry->history.append(history_t(entry->event,entry->start,now));
#endif
    entry->local += now-entry->start;
    entry->start = wall_time_t(0);
    entry = 0;
  }
}

#endif

/****************** thread_pool_t *****************/

namespace {

typedef std::unique_lock<std::mutex> lock_t;

class thread_pool_t : private boost::noncopyable {
public:
  const thread_type_t type;
  const int count;
private:
  vector<pthread_t> threads;
#if BLOCKING
  std::mutex mutex;
  std::condition_variable master_cond, worker_cond;
#else
  spinlock_t spinlock;
  // In nonblocking mode, we busy wait outside of the spinlock to reduce starvation possibilities.  It is
  // probably fine to access jobs.size() outside of a spinlock, but I'm not sure, so maintain our own count.
  volatile int job_count;
#endif
  deque<function<void()>> jobs;
  std::exception_ptr error;
  volatile int waiting; // Number of threads waiting for jobs to run
  volatile bool die;

  friend void pentago::threads_wait_all();
  friend void pentago::threads_wait_all_help();

public:
  thread_pool_t(thread_type_t type, int count, int delta_priority);
  ~thread_pool_t();

  void wait(); // Wait for all jobs to complete
  void schedule(function<void()>&& f, bool soon=false); // Schedule a job

private:
  static void* worker(void* pool);
  void shutdown();
  void set_die();
};

thread_pool_t::thread_pool_t(thread_type_t type, int count, int delta_priority)
  : type(type)
  , count(count)
#if !BLOCKING
  , job_count(0)
#endif
  , waiting(0)
  , die(false) {
  GEODE_ASSERT(count>0);

  // Prepare to create threads
  pthread_attr_t attr;
  CHECK(pthread_attr_init(&attr));

  // Adjust priority
#if !defined(__bgq__)
  sched_param sched;
  CHECK(pthread_attr_getschedparam(&attr,&sched));
  int policy;
  CHECK(pthread_attr_getschedpolicy(&attr,&policy));
  int old_priority = sched.sched_priority;
  Box<int> range(sched_get_priority_min(policy),sched_get_priority_max(policy));
  sched.sched_priority = range.clamp(old_priority+delta_priority);
  slog("%s thread pool: threads = %d, priority = %d",
       type==CPU ? "cpu" : type==IO ? "io" : "<unknown>", count, sched.sched_priority);
  CHECK(pthread_attr_setschedparam(&attr,&sched));
#else
  // Thread priority setting doesn't seem to work on BlueGene
  slog("%s thread pool: threads = %d, priority = unchanged",
       type==CPU ? "cpu" : type==IO ? "io" : "<unknown>", count);
#endif

  // Create threads
  for (int id=0;id<count;id++) {
    pthread_t thread;
    int r = pthread_create(&thread,&attr,&thread_pool_t::worker,(void*)this);
    if (!r)
      threads.push_back(thread);
    else {
      pthread_attr_destroy(&attr);
      shutdown();
      THROW(RuntimeError,"thread_pool_t: thread creation failed, %s",strerror(r));
    }
  }
  pthread_attr_destroy(&attr);
}

thread_pool_t::~thread_pool_t() {
  shutdown();
}

void thread_pool_t::set_die() {
#if BLOCKING
  lock_t lock(mutex);
#else
  spin_t spin(spinlock);
#endif
  die = true;
  // Enforce the following invariant: if die = true, there are no jobs available
  jobs.clear();
#if BLOCKING
  worker_cond.notify_all();
  master_cond.notify_all();
#else
  job_count = 0;
#endif
}

void thread_pool_t::shutdown() {
  set_die();
  for (pthread_t& thread : threads)
    CHECK(pthread_join(thread,0));
}

void* thread_pool_t::worker(void* pool_) {
  thread_pool_t& pool = *(thread_pool_t*)pool_;
  time_info.init_thread(pool.type);
  const time_kind_t idle = pool.type==CPU?cpu_idle_kind:pool.type==IO?io_idle_kind:_time_kinds;
  GEODE_ASSERT(idle!=_time_kinds);
  for (;;) {
    // Grab a job
    function<void()> f;
    {
      thread_time_t time(idle,unevent);
#if BLOCKING
      // Blocking version using pthread mutexes and condition variables
      lock_t lock(pool.mutex);
      for (;;) {
        if (pool.jobs.size()) {
          f = std::move(pool.jobs.front());
          pool.jobs.pop_front();
          break;
        } else if (pool.die) {
          return 0;
        } else {
          pool.master_cond.notify_one();
          pool.waiting++;
          pool.worker_cond.wait(lock);
          pool.waiting--;
        }
      }
#else
      // Nonblocking version using spinlocks
      {
        // Try once and increment waiting if unsuccessful
        spin_t spin(pool.spinlock);
        if (pool.job_count) {
          f = std::move(pool.jobs.front());
          pool.jobs.pop_front();
          pool.job_count--;
        } else
          pool.waiting++;
      }
      if (!f) {
        // Otherwise spin until a job is found or we die.  In either case decrement waiting.
        for (;;) {
          // Spin without locking until something might be available.  In the common case
          // where all worker threads wait for a significant while, this allows the master
          // thread to snap up the spinlock without fighting against idle workings.
          while (!pool.job_count && !pool.die);
          // Lock and see if the job is still there
          spin_t spin(pool.spinlock);
          if (pool.job_count) {
            f = std::move(pool.jobs.front());
            pool.jobs.pop_front();
            pool.job_count--;
            pool.waiting--;
            break;
          } else if (pool.die) {
            pool.waiting--;
            return 0;
          }
        }
      }
#endif
    }

    // Run the job
    try {
      f();
    } catch (const exception& e) {
      if (throw_callback)
        throw_callback(e.what());
      {
#if BLOCKING
        lock_t lock(pool.mutex);
#else
        spin_t spin(pool.spinlock);
#endif
        if (!pool.error)
          pool.error = std::current_exception();
      }
      pool.set_die();
      return 0;
    }
  }
}

void thread_pool_t::schedule(function<void()>&& f, bool soon) {
  GEODE_ASSERT(threads.size());
#if BLOCKING
  lock_t lock(mutex);
#else
  spin_t spin(spinlock);
#endif
  if (error)
    std::rethrow_exception(error);
  GEODE_ASSERT(!die);
  if (soon)
    jobs.push_front(std::move(f));
  else
    jobs.push_back(std::move(f));
#if BLOCKING
  if (waiting)
    worker_cond.notify_one();
#else
  job_count++;
#endif
}

void thread_pool_t::wait() {
  thread_time_t time(master_idle_kind,unevent);
#if BLOCKING
  lock_t lock(mutex);
  while (!die && (jobs.size() || waiting<count))
    master_cond.wait(lock);
  if (error)
    std::rethrow_exception(error);
  GEODE_ASSERT(!die);
#else
  for (;;) {
    while (!die && (job_count || waiting<count));
    spin_t spin(spinlock);
    if (!(!die && (job_count || waiting<count))) {
      if (error)
        std::rethrow_exception(error);
      GEODE_ASSERT(!die);
      break;
    }
  }
#endif
}

unique_ptr<thread_pool_t> cpu_pool;
unique_ptr<thread_pool_t> io_pool;

}

unit_t init_threads(int cpu_threads, int io_threads) {
  if (cpu_threads!=-1 || io_threads!=-1 || !cpu_pool) {
    GEODE_ASSERT(!cpu_pool && !io_pool);
    init_papi();
    time_info.init_thread(MASTER);
    if (cpu_threads<0)
      cpu_threads = int(sysconf(_SC_NPROCESSORS_ONLN));
    if (io_threads<0)
      io_threads = 2;
    if (cpu_threads)
      cpu_pool.reset(new thread_pool_t(CPU,cpu_threads,0));
    if (io_threads)
      io_pool.reset(new thread_pool_t(IO,io_threads,1000));
    time_info.local_start = wall_time();
  }
  return unit;
}

Vector<int,2> thread_counts() {
  return vec(cpu_pool ? cpu_pool->count : 0, io_pool ? io_pool->count : 0);
}

void threads_schedule(thread_type_t type, function<void()>&& f, bool soon) {
  GEODE_ASSERT(type==CPU || type==IO);
  const auto& pool = type == CPU ? cpu_pool : io_pool;
  GEODE_ASSERT(pool);
  pool->schedule(std::move(f), soon);
}

void threads_wait_all() {
  if (!io_pool)
    cpu_pool->wait();
  else
    for (;;) {
      cpu_pool->wait();
      io_pool->wait();
#if BLOCKING
      lock_t cpu_lock(cpu_pool->mutex);
      lock_t io_lock(io_pool->mutex);
#else
      spin_t cpu_spin(cpu_pool->spinlock);
      spin_t io_spin(io_pool->spinlock);
#endif
      if (cpu_pool->error)
        std::rethrow_exception(cpu_pool->error);
      if (io_pool->error)
        std::rethrow_exception(io_pool->error);
      GEODE_ASSERT(!cpu_pool->die);
      GEODE_ASSERT(!io_pool->die);
      if (   !cpu_pool->jobs.size() && cpu_pool->waiting==cpu_pool->count
          && !io_pool->jobs.size() && io_pool->waiting==io_pool->count)
        break;
    }
}

// Note: This function doesn't account for the case when the pool temporarily
// runs out of jobs, but currently running jobs schedule a bunch more.  If
// that happens, it safely but stupidly reverts to wait().
void threads_wait_all_help() {
  auto& pool = *cpu_pool;
  for (;;) {
    // Grab a job
    function<void()> f;
    {
#if BLOCKING
      lock_t lock(pool.mutex);
#else
      spin_t spin(pool.spinlock);
#endif
      if (pool.die || !pool.jobs.size())
        goto wait;
      else {
        f = std::move(pool.jobs.front());
        pool.jobs.pop_front();
#if !BLOCKING
        pool.job_count--;
#endif
      }
    }

    // Run the job
    try {
      f();
    } catch (const exception& e) {
      if (throw_callback)
        throw_callback(e.what());
      {
#if BLOCKING
        lock_t lock(pool.mutex);
#else
        spin_t spin(pool.spinlock);
#endif
        if (!pool.error)
          pool.error = std::current_exception();
      }
      pool.set_die();
      goto wait;
    }
  }
  wait:
  threads_wait_all();
}

/****************** time reports *****************/

thread_times_t clear_thread_times() {
  threads_wait_all();
  spin_t spin(time_lock);
  wall_time_t now = wall_time();
  Array<wall_time_t> times(_time_kinds);
  Array<papi_t,2> papi(_time_kinds, papi_count);
  for (auto table : time_tables) {
    // Account for any active timers (which should all be idle)
    for (int k : range((int)master_missing_kind)) {
      auto& entry = table->times[k];
      if (entry.start) {
#if HISTORY
        entry.history.append(history_t(entry.event,entry.start,now));
#endif
        entry.local += now-entry.start;
        entry.start = now;
      }
    }
    // Compute missing time
    wall_time_t missing = now-time_info.local_start;
    for (int k : range((int)master_missing_kind))
      missing -= table->times[k].local;
    const int missing_kind = master_missing_kind+table->type;
    times[missing_kind] += missing;
    table->times[missing_kind].total += missing;
    // Clear local times
    for (int k : range((int)master_missing_kind)) {
      auto& entry = table->times[k];
      times[k] += entry.local;
      entry.total += entry.local;
      entry.local = wall_time_t(0);
#if PAPI_MAX
      for (const int c : range(papi_count)) {
        auto& local = entry.papi_local[c];
        entry.papi_total[c] += local;
        papi(k,c) += local;
        local = 0;
      }
#endif
    }
  }
  time_info.local_start = now;
  return thread_times_t({times, papi});
}

thread_times_t total_thread_times() {
  clear_thread_times();
  spin_t spin(time_lock);
  Array<wall_time_t> times(_time_kinds);
  Array<papi_t,2> papi(_time_kinds,papi_count);
  for (const auto table : time_tables)
    for (const int k : range((int)_time_kinds)) {
      times[k] += table->times[k].total;
#if PAPI_MAX
      for (const int p : range(papi_count))
        papi(k,p) += table->times[k].papi_total[p];
#endif 
    }
  return thread_times_t({times, papi});
}

vector<const char*> time_kind_names() {
  vector<const char*> names(master_missing_kind);
  #define FIELD(kind) names[kind##_kind] = #kind;
  FIELD(compress)
  FIELD(decompress)
  FIELD(snappy)
  FIELD(unsnappy)
  FIELD(filter)
  FIELD(copy)
  FIELD(schedule)
  FIELD(wait)
  FIELD(mpi)
  FIELD(partition)
  FIELD(compute)
  FIELD(accumulate)
  FIELD(count)
  FIELD(meaningless)
  FIELD(read)
  FIELD(write)
  FIELD(write_sections)
  FIELD(write_counts)
  FIELD(write_sparse)
  FIELD(allocate_line)
  FIELD(request_send)
  FIELD(response_send)
  FIELD(response_recv)
  FIELD(wakeup)
  FIELD(output_send)
  FIELD(output_recv)
  FIELD(compacting)
  FIELD(master_idle)
  FIELD(cpu_idle)
  FIELD(io_idle)
  for (const int k : range((int)master_missing_kind))
    GEODE_ASSERT(names[k],format("missing name for kind %d",k));
  return names;
}

void report_thread_times(RawArray<const wall_time_t> times, const string& name) {
  GEODE_ASSERT(times.size()==_time_kinds);
  vector<const char*> names = time_kind_names();

  // Print times
  slog("timing %s", name);
  for (const int k : range((int)master_missing_kind))
    if (times[k])
      slog("  %-20s %10.4f s", names[k], times[k].seconds());
  if (io_pool)
    slog("  missing: master %.4f, cpu %.4f, io %.4f", times[master_missing_kind].seconds(),
         times[cpu_missing_kind].seconds(), times[io_missing_kind].seconds());
  else
    slog("  missing: master %.4f, cpu %.4f", times[master_missing_kind].seconds(),
         times[cpu_missing_kind].seconds());
  slog("  total %.4f", times.sum().seconds());
}

void report_papi_counts(RawArray<const papi_t,2> papi) {
  GEODE_ASSERT(papi.shape() == vec(_time_kinds, papi_count));
#if PAPI_MAX
  if (!papi_count)
    return;
  vector<const char*> names = time_kind_names();

  // Print times
  {
    string s = format("%-22s", "papi");
    for (const int p : range(papi_count))
      s += format(" %14s", papi_event_name(papi_events[p]));
    slog(s);
  }
  for (const int k : range((int)master_idle_kind))
    if (!papi[k].contains_only(0)) {
      string s = format("  %-20s", names[k]);
      for (const int p : range(papi_count))
        s += format(" %14lld", papi(k,p));
      slog(s);
    }
#endif
}

bool thread_history_enabled() {
  return HISTORY;
}

vector<vector<Array<const history_t>>> thread_history() {
  clear_thread_times();
  spin_t spin(time_lock);
  vector<vector<Array<const history_t>>> data(time_tables.size());
#if HISTORY
  for (int t : range((int)data.size())) {
    auto& table = *time_tables[t];
    for (int k : range((int)master_missing_kind))
      data[t].push_back(table.times[k].history.const_());
  }
#endif
  return data;
}

void write_thread_history(const string& filename) {
#if HISTORY
  typedef Vector<int64_t,3> Elem;
  static_assert(sizeof(Elem)==sizeof(history_t),"");
  const auto history = thread_history();
  if (!history.size())
    return;
  FILE* file = fopen(filename.c_str(),"w");
  GEODE_ASSERT(file);
  int offset = 1+history.size()*history[0].size();
  Array<Elem> header;
  header.append(Elem(history.size(),history[0].size(),0));
  for (const auto& thread : history)
    for (const auto& trace : thread) {
      header.append(Elem(offset,offset+trace.size(),0));
      offset += trace.size();
    }
  fwrite(header.data(),sizeof(Elem),header.size(),file);
  for (const auto& thread : history)
    for (const auto& trace : thread)
      fwrite(trace.data(),sizeof(Elem),trace.size(),file);
  fclose(file); 
#endif
}

}
