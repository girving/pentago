// Thread utilities

#include "thread.h"
#include <other/core/python/Class.h>
#include <other/core/random/counter.h>
#include <other/core/structure/HashtableIterator.h>
#include <other/core/structure/Tuple.h>
#include <other/core/utility/Log.h>
#include <other/core/vector/Interval.h>
#include <other/core/utility/process.h>
#include <boost/bind.hpp>
#include <set>
namespace pentago {

using Log::cout;
using std::endl;
using std::flush;
using std::set;
using std::exception;

#if !OTHER_THREAD_SAFE
#error "pentago requires thread_safe=1"
#endif

#define CHECK(exp) ({ \
  int r_ = (exp); \
  if (r_) \
    throw RuntimeError(format("thread_pool_t: %s failed, %s",#exp,strerror(r_))); \
  })

static pthread_t master;

static bool is_master() {
  return pthread_equal(master,pthread_self());
}

/****************** thread_time_t *****************/

struct time_entry_t {
  int id;
  double total, local, start;
  time_entry_t() {}
  time_entry_t(int id) : id(id), total(0), local(0), start(0) {}
};

typedef Tuple<thread_type_t,Hashtable<const void*,time_entry_t>> time_table_t;
vector<time_table_t*> time_tables;
static mutex_t time_mutex;

struct time_info_t {
  pthread_key_t key;
  double total_start, local_start;

  time_info_t() {
    pthread_key_create(&key,0);
    total_start = local_start = 0;
  }

  void init_thread(thread_type_t type) {
    if (!pthread_getspecific(key)) {
      lock_t lock(time_mutex);
      time_tables.push_back(new time_table_t(type,Hashtable<const void*,time_entry_t>()));
      pthread_setspecific(key,time_tables.back());
    }
  }
};
static time_info_t time_info;

thread_type_t thread_type() {
  auto table = (time_table_t*)pthread_getspecific(time_info.key);
  OTHER_ASSERT(table);
  return table->x;
}

static inline time_entry_t& time_entry(const char* name) {
  auto table = (time_table_t*)pthread_getspecific(time_info.key);
  OTHER_ASSERT(table);
  time_entry_t* entry = table->y.get_pointer((void*)name);
  if (entry)
    return *entry;
  table->y.set((void*)name,time_entry_t(table->y.size()));
  return table->y.get((void*)name);
}

static inline double time() {
  timeval tv;
  gettimeofday(&tv,0);
  return (double)tv.tv_sec+1e-6*tv.tv_usec;
}

thread_time_t::thread_time_t(const char* name)
  : entry(time_entry(name)) {
  entry.start = time();
}

thread_time_t::~thread_time_t() {
  entry.local += time()-entry.start;
  entry.start = 0;
}

/****************** thread_pool_t *****************/

namespace {

class thread_pool_t : public Object {
public:
  OTHER_DECLARE_TYPE

  const thread_type_t type;
  const int count;
  const string idle_name;
private:
  vector<pthread_t> threads;
  mutex_t mutex;
  cond_t master_cond, worker_cond;
  vector<function<void()>> jobs;
  ExceptionValue error;
  int waiting;
  bool die;

  friend void pentago::wait_all();

  thread_pool_t(thread_type_t type, int threads, int delta_priority);
public:
  ~thread_pool_t();

  void wait(); // wait for all jobs to complete
  void schedule(const function<void()>& f); // schedule a job

private:
  static void* worker(void* pool);
  void shutdown();
};

OTHER_DEFINE_TYPE(thread_pool_t)

thread_pool_t::thread_pool_t(thread_type_t type, int count, int delta_priority)
  : type(type)
  , count(count)
  , idle_name(type==CPU?"cpu-idle":type==IO?"io-idle":"unknown-idle")
  , master_cond(mutex)
  , worker_cond(mutex)
  , waiting(0)
  , die(false) {
  OTHER_ASSERT(count>0);

  // Adjust priority
  pthread_attr_t attr;
  sched_param sched;
  CHECK(pthread_attr_init(&attr));
  CHECK(pthread_attr_getschedparam(&attr,&sched));
  int policy;
  CHECK(pthread_attr_getschedpolicy(&attr,&policy));
  int old_priority = sched.sched_priority;
  Interval<int> range(sched_get_priority_min(policy),sched_get_priority_max(policy));
  sched.sched_priority = range.clamp(old_priority+delta_priority);
  cout << (type==CPU?"cpu":type==IO?"io":"<unknown>")<<" thread pool: threads = "<<count<<", priority = "<<sched.sched_priority<<endl;
  CHECK(pthread_attr_setschedparam(&attr,&sched));

  // Create threads
  for (int id=0;id<count;id++) {
    pthread_t thread;
    int r = pthread_create(&thread,&attr,&thread_pool_t::worker,(void*)this);
    if (!r)
      threads.push_back(thread);
    else {
      pthread_attr_destroy(&attr);
      shutdown();
      throw RuntimeError(format("thread_pool_t: thread creation failed, %s",strerror(r)));
    }
  }
  pthread_attr_destroy(&attr);
}

thread_pool_t::~thread_pool_t() {
  shutdown();
}

void thread_pool_t::shutdown() {
  {
    lock_t lock(mutex);
    jobs.clear();
    die = true;
    worker_cond.broadcast();
  }
  for (pthread_t& thread : threads)
    CHECK(pthread_join(thread,0));
}

void* thread_pool_t::worker(void* pool_) {
  thread_pool_t& pool = *(thread_pool_t*)pool_;
  time_info.init_thread(pool.type);
  for (;;) {
    // Grab a job
    bool die = false;
    function<void()> f;
    {
      thread_time_t time(pool.idle_name.c_str());
      lock_t lock(pool.mutex);
      while (!die && !f) {
        if (pool.die)
          die = true;
        else if (pool.jobs.size()) {
          swap(f,pool.jobs.back());
          pool.jobs.pop_back();
        } else {
          pool.master_cond.signal();
          pool.waiting++;
          pool.worker_cond.wait();
          pool.waiting--;
        }
      }
    }

    // Were we killed?
    if (die)
      return 0;

    // Run the job
    try {
      f();
    } catch (const exception& e) {
      lock_t lock(pool.mutex);
      pool.error = e;
      pool.die = true;
      return 0;
    }
  }
}

void thread_pool_t::schedule(const function<void()>& f) {
  lock_t lock(mutex);
  if (error)
    error.throw_();
  jobs.push_back(f);
  if (waiting)
    worker_cond.signal();
}

void thread_pool_t::wait() {
  OTHER_ASSERT(is_master());
  thread_time_t time("master-idle");
  lock_t lock(mutex);
  while (jobs.size() || waiting<count)
    master_cond.wait();
}

Ptr<thread_pool_t> cpu_pool;
Ptr<thread_pool_t> io_pool;

}

void init_thread_pools(int cpu_threads, int io_threads) {
  if (cpu_threads<0)
    cpu_threads = sysconf(_SC_NPROCESSORS_ONLN);
  if (io_threads<0)
    io_threads = 2;
  OTHER_ASSERT(!!cpu_pool == !!io_pool);
  if (cpu_pool)
    OTHER_ASSERT(cpu_pool->count==cpu_threads && io_pool->count==io_threads);
  else {
    cpu_pool = new_<thread_pool_t>(CPU,cpu_threads,0);
    io_pool = new_<thread_pool_t>(IO,io_threads,1000);
    time_info.total_start = time_info.local_start = time();
  }
}

void schedule(thread_type_t type, const function<void()>& f) {
  OTHER_ASSERT(type==CPU || type==IO);
  (type==CPU?cpu_pool:io_pool)->schedule(f);
}

void wait_all() {
  for (;;) {
    cpu_pool->wait();
    io_pool->wait();
    lock_t cpu_lock(cpu_pool->mutex);
    lock_t io_lock(io_pool->mutex);
    if (   !cpu_pool->jobs.size() && cpu_pool->waiting==cpu_pool->count
        && !io_pool->jobs.size() && io_pool->waiting==io_pool->count)
      break;
  }
}

/****************** time reports *****************/

void clear_thread_times() {
  OTHER_ASSERT(is_master());
  wait_all();
  lock_t lock(time_mutex);
  double now = time();
  for (auto table : time_tables)
    for (HashtableIterator<const void*,time_entry_t> it(table->y);it.valid();it.next()) {
      time_entry_t& entry = it.data();
      if (entry.start) {
        entry.local += now-entry.start;
        entry.start = now;
      }
      entry.total += entry.local;
      entry.local = 0;
    }
}

static const char* sanitize_name(const string& name) {
  return name.size()?name.c_str():"<empty>";
}

void report_thread_times(bool total) {
  OTHER_ASSERT(is_master());
  wait_all();
  lock_t lock(time_mutex);
  double totals[3] = {0,0,0};
  const double now = time(),
               start = total?time_info.total_start:time_info.local_start,
               elapsed = now-start;
  OTHER_ASSERT(start);
  time_info.local_start = now;
  Hashtable<string,Tuple<int,double>> times;
  Hashtable<string> type_to_name[3];
  for (auto table : time_tables)
    for (HashtableIterator<const void*,time_entry_t> it(table->y);it.valid();it.next()) {
      const string name = (const char*)it.key();
      time_entry_t& entry = it.data();
      if (total)
        type_to_name[table->x].set(name);
      if (entry.start) {
        entry.local += now-entry.start;
        entry.start = now;
      }
      auto& time = times.get_or_insert(name,tuple(10000,0.));
      time.x = min(time.x,entry.id);
      entry.total += entry.local;
      double t = total?entry.total:entry.local;
      entry.local = 0;
      time.y += t;
      totals[table->x] += t;
    }

  // Sort by id
  set<Tuple<int,string>> order;
  for (HashtableIterator<string,Tuple<int,double>> it(times);it.valid();it.next())
    order.insert(tuple(it.data().x,it.key()));

  // Print times
  cout << "timing\n";
  for (auto& iname : order)
    cout << format("  %-20s %8.4f s\n",sanitize_name(iname.y.c_str()),times.get(iname.y.c_str()).y);
  cout << format("  missing: master %.4f, cpu %.4f, io %.4f\n",elapsed-totals[MASTER],cpu_pool->count*elapsed-totals[CPU],io_pool->count*elapsed-totals[IO]);
  cout << flush;

  // Show which jobs ran on which type of thread
  if (total) {
    cout << "job types\n";
    for (int i=0;i<3;i++)
      if (type_to_name[i].size()) {
        cout << (i==MASTER?"  master = ":i==CPU?"  cpu = ":i==IO?"  io = ":"  ? = ");
        for (HashtableIterator<string> it(type_to_name[i]);it.valid();it.next())
          cout << sanitize_name(it.key())<<' ';
        cout << endl;
      }
  }
}

/****************** testing *****************/

static void add_noise(Array<uint128_t> data, int key, mutex_t* mutex) {
  for (int i=0;i<data.size();i++) {
    lock_t lock(*mutex);
    data[i] += threefry(key,i);
  }
  if (thread_type()==CPU)
    io_pool->schedule(boost::bind(add_noise,data,key+1,mutex));
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
    Array<uint128_t> parallel(serial.size());
    for (int i=0;i<jobs;i++)
      cpu_pool->schedule(boost::bind(add_noise,parallel,2*i,&mutex));
    wait_all();

    // Compare
    OTHER_ASSERT(serial==parallel);
  }
}

}
using namespace pentago;

void wrap_thread() {
  master = pthread_self();
  time_info.init_thread(MASTER);
  OTHER_FUNCTION(init_thread_pools)
  OTHER_FUNCTION(thread_pool_test)
  OTHER_FUNCTION(clear_thread_times)
  OTHER_FUNCTION(report_thread_times)
}
