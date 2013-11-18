// Thread utilities

#include <pentago/utility/thread.h>
#include <pentago/utility/debug.h>
#include <pentago/utility/spinlock.h>
#include <pentago/utility/wall_time.h>
#include <geode/python/Class.h>
#include <geode/python/enum.h>
#include <geode/python/stl.h>
#include <geode/random/counter.h>
#include <geode/structure/Tuple.h>
#include <geode/utility/curry.h>
#include <geode/utility/Log.h>
#include <geode/utility/process.h>
#include <geode/utility/stl.h>
#include <geode/geometry/Box.h>
#include <stdio.h>
#include <deque>
#include <set>
namespace geode {
GEODE_DEFINE_ENUM(thread_type_t,GEODE_EXPORT)
}
namespace pentago {


/****************** Configuration *****************/

// Flip to enable history tracking
#define HISTORY 0

// Whether to use blocking pthread locking or nonblocking spinlocks.
// Use nonblocking mode (0) except for testing purposes.
#define BLOCKING 0

#if !GEODE_THREAD_SAFE
#error "pentago requires thread_safe=1"
#endif

/********************** Setup *********************/

using Log::cout;
using std::endl;
using std::flush;
using std::deque;
using std::set;
using std::exception;

static pthread_t master;

static bool is_master() {
  return pthread_equal(master,pthread_self());
}

#define CHECK(exp) ({ \
  int r_ = (exp); \
  if (r_) \
    THROW(RuntimeError,"thread_pool_t: %s failed, %s",#exp,strerror(r_)); \
  })

/****************** thread_time_t *****************/

struct time_entry_t {
  wall_time_t total, local, start;
  event_t event;
#if HISTORY
  Array<history_t> history;
#endif

  time_entry_t()
    : total(0), local(0), start(0), event(0) {}
};

struct time_table_t {
  thread_type_t type;
  time_entry_t times[_time_kinds];
};
vector<time_table_t*> time_tables;
static spinlock_t time_lock;

struct time_info_t {
  pthread_key_t key;
  wall_time_t total_start, local_start;

  time_info_t()
    : total_start(0), local_start(0) {
    pthread_key_create(&key,0);
  }

  void init_thread(thread_type_t type) {
    if (!pthread_getspecific(key)) {
      spin_t spin(time_lock);
      auto table = new time_table_t;
      table->type = type;
      time_tables.push_back(table);
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
}

thread_time_t::~thread_time_t() {
  stop();
}

void thread_time_t::stop() {
  if (entry) {
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

/****************** pthread locking *****************/

namespace {

struct mutex_t : public boost::noncopyable {
  pthread_mutex_t mutex;

  mutex_t() {
    GEODE_ASSERT(!pthread_mutex_init(&mutex,0));
  }

  ~mutex_t() {
    pthread_mutex_destroy(&mutex);
  }
};

struct lock_t : public boost::noncopyable {
  mutex_t& mutex;

  lock_t(mutex_t& mutex)
    : mutex(mutex) {
    pthread_mutex_lock(&mutex.mutex);
  }

  ~lock_t() {
    pthread_mutex_unlock(&mutex.mutex);
  }
};

#if BLOCKING
struct cond_t : public boost::noncopyable {
  mutex_t& mutex;
  pthread_cond_t cond;

  cond_t(mutex_t& mutex)
    : mutex(mutex) {
    GEODE_ASSERT(!pthread_cond_init(&cond,0));
  }

  ~cond_t() {
    pthread_cond_destroy(&cond);
  }

  void broadcast() {
    pthread_cond_broadcast(&cond);
  }

  void signal() {
    pthread_cond_signal(&cond);
  }

  void wait() {
    pthread_cond_wait(&cond,&mutex.mutex);
  }
};
#endif

}

/****************** thread_pool_t *****************/

namespace {

class thread_pool_t : public Object {
public:
  GEODE_DECLARE_TYPE(GEODE_NO_EXPORT)

  const thread_type_t type;
  const int count;
private:
  vector<pthread_t> threads;
#if BLOCKING
  mutex_t mutex;
  cond_t master_cond, worker_cond;
#else
  spinlock_t spinlock;
  // In nonblocking mode, we busy wait outside of the spinlock to reduce starvation possibilities.  It is
  // probably fine to access jobs.size() outside of a spinlock, but I'm not sure, so maintain our own count.
  volatile int job_count;
#endif
  deque<job_base_t*> jobs;
  ExceptionValue error;
  volatile int waiting; // Number of threads waiting for jobs to run
  volatile bool die;

  friend void pentago::threads_wait_all();
  friend void pentago::threads_wait_all_help();

  thread_pool_t(thread_type_t type, int count, int delta_priority);
public:
  ~thread_pool_t();

  void wait(); // Wait for all jobs to complete
  void schedule(job_t&& f, bool soon=false); // Schedule a job

private:
  static void* worker(void* pool);
  void shutdown();
  void set_die();
};

GEODE_DEFINE_TYPE(thread_pool_t)

thread_pool_t::thread_pool_t(thread_type_t type, int count, int delta_priority)
  : type(type)
  , count(count)
#if BLOCKING
  , master_cond(mutex)
  , worker_cond(mutex)
#else
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
  cout << (type==CPU?"cpu":type==IO?"io":"<unknown>")<<" thread pool: threads = "<<count<<", priority = "<<sched.sched_priority<<endl;
  CHECK(pthread_attr_setschedparam(&attr,&sched));
#else
  // Thread priority setting doesn't seem to work on BlueGene
  cout << (type==CPU?"cpu":type==IO?"io":"<unknown>")<<" thread pool: threads = "<<count<<", priority = <unchanged>"<<endl;
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
  for (auto job : jobs)
    delete job;
  jobs.clear();
#if BLOCKING
  worker_cond.broadcast();
  master_cond.broadcast();
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
    job_t f;
    {
      thread_time_t time(idle,unevent);
#if BLOCKING
      // Blocking version using pthread mutexes and condition variables
      lock_t lock(pool.mutex);
      for (;;) {
        if (pool.jobs.size()) {
          f.reset(pool.jobs.front());
          pool.jobs.pop_front();
          break;
        } else if (pool.die) {
          return 0;
        } else {
          pool.master_cond.signal();
          pool.waiting++;
          pool.worker_cond.wait();
          pool.waiting--;
        }
      }
#else
      // Nonblocking version using spinlocks
      {
        // Try once and increment waiting if unsuccessful
        spin_t spin(pool.spinlock);
        if (pool.job_count) {
          f.reset(pool.jobs.front());
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
            f.reset(pool.jobs.front());
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
          pool.error = e;
      }
      pool.set_die();
      return 0;
    }
  }
}

void thread_pool_t::schedule(job_t&& f, bool soon) {
  GEODE_ASSERT(threads.size());
#if BLOCKING
  lock_t lock(mutex);
#else
  spin_t spin(spinlock);
#endif
  if (error)
    error.throw_();
  GEODE_ASSERT(!die);
  if (soon)
    jobs.push_front(f.release());
  else
    jobs.push_back(f.release());
#if BLOCKING
  if (waiting)
    worker_cond.signal();
#else
  job_count++;
#endif
}

void thread_pool_t::wait() {
  GEODE_ASSERT(is_master());
  thread_time_t time(master_idle_kind,unevent);
#if BLOCKING
  lock_t lock(mutex);
  while (!die && (jobs.size() || waiting<count))
    master_cond.wait();
  if (error)
    error.throw_();
  GEODE_ASSERT(!die);
#else
  for (;;) {
    while (!die && (job_count || waiting<count));
    spin_t spin(spinlock);
    if (!(!die && (job_count || waiting<count))) {
      if (error)
        error.throw_();
      GEODE_ASSERT(!die);
      break;
    }
  }
#endif
}

Ptr<thread_pool_t> cpu_pool;
Ptr<thread_pool_t> io_pool;

}

void init_threads(int cpu_threads, int io_threads) {
  GEODE_ASSERT(cpu_threads);
  if (cpu_threads==-1 && io_threads==-1 && cpu_pool)
    return;
  GEODE_ASSERT(!cpu_pool && !io_pool);
  master = pthread_self();
  time_info.init_thread(MASTER);
  if (cpu_threads<0)
    cpu_threads = sysconf(_SC_NPROCESSORS_ONLN);
  if (io_threads<0)
    io_threads = 2;
  cpu_pool = new_<thread_pool_t>(CPU,cpu_threads,0);
  if (io_threads)
    io_pool = new_<thread_pool_t>(IO,io_threads,1000);
  time_info.total_start = time_info.local_start = wall_time();
}

Vector<int,2> thread_counts() {
  return vec(cpu_pool?cpu_pool->count:0,
              io_pool? io_pool->count:0);
}

void threads_schedule(thread_type_t type, job_t&& f, bool soon) {
  GEODE_ASSERT(type==CPU || type==IO);
  if (type!=CPU) GEODE_ASSERT(io_pool);
  (type==CPU?cpu_pool:io_pool)->schedule(geode::move(f),soon);
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
        cpu_pool->error.throw_();
      if (io_pool->error)
        io_pool->error.throw_();
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
    job_t f;
    {
#if BLOCKING
      lock_t lock(pool.mutex);
#else
      spin_t spin(pool.spinlock);
#endif
      if (pool.die || !pool.jobs.size())
        goto wait;
      else {
        f.reset(pool.jobs.front());
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
          pool.error = e;
      }
      pool.set_die();
      goto wait;
    }
  }
  wait:
  threads_wait_all();
}

/****************** time reports *****************/

Array<wall_time_t> clear_thread_times() {
  GEODE_ASSERT(is_master());
  threads_wait_all();
  spin_t spin(time_lock);
  wall_time_t now = wall_time();
  Array<wall_time_t> result(_time_kinds);
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
    result[missing_kind] += missing;
    table->times[missing_kind].total += missing;
    // Clear local times
    for (int k : range((int)master_missing_kind)) {
      auto& entry = table->times[k];
      result[k] += entry.local;
      entry.total += entry.local;
      entry.local = wall_time_t(0);
    }
  }
  time_info.local_start = now;
  return result;
}

Array<wall_time_t> total_thread_times() {
  clear_thread_times();
  spin_t spin(time_lock);
  Array<wall_time_t> result(_time_kinds);
  for (auto table : time_tables)
    for (int k : range((int)_time_kinds))
      result[k] += table->times[k].total;
  return result;
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
  for (int k : range((int)master_missing_kind))
    GEODE_ASSERT(names[k],format("missing name for kind %d",k));
  return names;
}

void report_thread_times(RawArray<const wall_time_t> times, const string& name) {
  GEODE_ASSERT(times.size()==_time_kinds);
  vector<const char*> names = time_kind_names();

  // Print times
  cout << "timing "<<name<<"\n";
  for (int k : range((int)master_missing_kind))
    if (times[k])
      cout << format("  %-20s %10.4f s\n",names[k],times[k].seconds());
  if (io_pool)
    cout << format("  missing: master %.4f, cpu %.4f, io %.4f\n",times[master_missing_kind].seconds(),times[cpu_missing_kind].seconds(),times[io_missing_kind].seconds());
  else
    cout << format("  missing: master %.4f, cpu %.4f\n",times[master_missing_kind].seconds(),times[cpu_missing_kind].seconds());
  cout << format("  total %.4f\n",times.sum().seconds());
  cout << flush;
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
  BOOST_STATIC_ASSERT(sizeof(Elem)==sizeof(history_t));
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

/****************** testing *****************/

static void add_noise(Array<uint128_t> data, int key, mutex_t* mutex, spinlock_t* spinlock) {
  const int n = data.size();
  for (int i=0;i<n/2;i++) {
    lock_t lock(*mutex);
    data[i] += threefry(key,i);
  }
  for (int i=n/2;i<n;i++) {
    spin_t spin(*spinlock);
    data[i] += threefry(key,i);
  }
  if (thread_type()==CPU)
    io_pool->schedule(curry(add_noise,data,key+1,mutex,spinlock));
}

static void thread_pool_test() {
  // Compute answers in serial
  const int jobs = 20;
  Array<uint128_t> serial(100);
  for (int i=0;i<2*jobs;i++)
    for (int j=0;j<serial.size();j++)
      serial[j] += threefry(i,j);

  for (int k=0;k<100;k++) {
    // Compute answers in parallel
    mutex_t mutex;
    spinlock_t spinlock;
    Array<uint128_t> parallel(serial.size());
    for (int i=0;i<jobs;i++)
      cpu_pool->schedule(curry(add_noise,parallel,2*i,&mutex,&spinlock));
    threads_wait_all();

    // Compare
    GEODE_ASSERT(serial==parallel);
  }
}

}
using namespace pentago;

void wrap_thread() {
  GEODE_FUNCTION(init_threads)
  GEODE_FUNCTION(thread_pool_test)
  GEODE_FUNCTION(clear_thread_times)
  GEODE_FUNCTION(total_thread_times)
  GEODE_FUNCTION(report_thread_times)
  GEODE_FUNCTION(thread_history)
  GEODE_FUNCTION(time_kind_names)

  GEODE_ENUM(thread_type_t)
  GEODE_ENUM_VALUE(MASTER)
  GEODE_ENUM_VALUE(CPU)
  GEODE_ENUM_VALUE(IO)
}
