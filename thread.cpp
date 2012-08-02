// Thread utilities

#include <pentago/thread.h>
#include <pentago/utility/debug.h>
#include <other/core/python/Class.h>
#include <other/core/python/stl.h>
#include <other/core/random/counter.h>
#include <other/core/structure/HashtableIterator.h>
#include <other/core/structure/Tuple.h>
#include <other/core/utility/Log.h>
#include <other/core/utility/process.h>
#include <other/core/utility/stl.h>
#include <other/core/vector/Interval.h>
#include <boost/bind.hpp>
#include <deque>
#include <set>
namespace pentago {

using Log::cout;
using std::endl;
using std::flush;
using std::deque;
using std::set;
using std::exception;

#if !OTHER_THREAD_SAFE
#error "pentago requires thread_safe=1"
#endif

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

static bool history = false;

struct time_entry_t {
  int id;
  double total, local, start;
  Array<Vector<double,2>> history;

  time_entry_t() {}
  time_entry_t(int id) : id(id), total(0), local(0), start(0) {}
};

struct time_table_t {
  thread_type_t type;
  int thread_id;
  Hashtable<const void*,time_entry_t> times;
};
vector<time_table_t*> time_tables;
static mutex_t time_mutex;

struct time_info_t {
  pthread_key_t key;
  int next_thread_id;
  double total_start, local_start;

  time_info_t() {
    pthread_key_create(&key,0);
    next_thread_id = 0;
    total_start = local_start = 0;
  }

  void init_thread(thread_type_t type) {
    if (!pthread_getspecific(key)) {
      lock_t lock(time_mutex);
      auto table = new time_table_t;
      table->type = type;
      table->thread_id = next_thread_id++;
      time_tables.push_back(table);
      pthread_setspecific(key,time_tables.back());
    }
  }
};
static time_info_t time_info;

thread_type_t thread_type() {
  auto table = (time_table_t*)pthread_getspecific(time_info.key);
  OTHER_ASSERT(table);
  return table->type;
}

static inline time_entry_t& time_entry(const char* name) {
  auto table = (time_table_t*)pthread_getspecific(time_info.key);
  OTHER_ASSERT(table);
  time_entry_t* entry = table->times.get_pointer((void*)name);
  if (entry)
    return *entry;
  table->times.set((void*)name,time_entry_t(table->times.size()));
  return table->times.get((void*)name);
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
  double now = time();
  if (history)
    entry.history.append(vec(entry.start,now));
  entry.local += now-entry.start;
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
  deque<function<void()>> jobs;
  ExceptionValue error;
  int waiting;
  bool die;

  friend void pentago::threads_wait_all();
  friend void pentago::threads_wait_all_help();

  thread_pool_t(thread_type_t type, int count, int delta_priority);
public:
  ~thread_pool_t();

  void wait(); // wait for all jobs to complete
  void schedule(const function<void()>& f); // schedule a job
  void schedule(const vector<function<void()>>& fs); // schedule many jobs

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
      THROW(RuntimeError,"thread_pool_t: thread creation failed, %s",strerror(r));
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
    function<void()> f;
    {
      thread_time_t time(pool.idle_name.c_str());
      lock_t lock(pool.mutex);
      while (!f) {
        if (pool.die)
          return 0;
        else if (pool.jobs.size()) {
          swap(f,pool.jobs.front());
          pool.jobs.pop_front();
        } else {
          pool.master_cond.signal();
          pool.waiting++;
          pool.worker_cond.wait();
          pool.waiting--;
        }
      }
    }

    // Run the job
    try {
      f();
    } catch (const exception& e) {
      if (throw_callback)
        throw_callback(e.what());
      lock_t lock(pool.mutex);
      if (!pool.error)
        pool.error = e;
      pool.die = true;
      pool.master_cond.signal();
      return 0;
    }
  }
}

void thread_pool_t::schedule(const function<void()>& f) {
  OTHER_ASSERT(threads.size());
  lock_t lock(mutex);
  if (error)
    error.throw_();
  OTHER_ASSERT(!die);
  jobs.push_back(f);
  if (waiting)
    worker_cond.signal();
}

void thread_pool_t::schedule(const vector<function<void()>>& fs) {
  OTHER_ASSERT(threads.size());
  lock_t lock(mutex);
  if (error)
    error.throw_();
  OTHER_ASSERT(!die);
  extend(jobs,fs);
  if (waiting)
    worker_cond.broadcast();
}

void thread_pool_t::wait() {
  OTHER_ASSERT(is_master());
  thread_time_t time("master-idle");
  lock_t lock(mutex);
  while (!die && (jobs.size() || waiting<count))
    master_cond.wait();
  if (error)
    error.throw_();
  OTHER_ASSERT(!die);
}

Ptr<thread_pool_t> cpu_pool;
Ptr<thread_pool_t> io_pool;

}

void set_thread_history(bool flag) {
  history = flag;
}

void init_threads(int cpu_threads, int io_threads) {
  OTHER_ASSERT(cpu_threads);
  if (cpu_threads==-1 && io_threads==-1 && cpu_pool)
    return;
  OTHER_ASSERT(!cpu_pool && !io_pool);
  master = pthread_self();
  time_info.init_thread(MASTER);
  if (cpu_threads<0)
    cpu_threads = sysconf(_SC_NPROCESSORS_ONLN);
  if (io_threads<0)
    io_threads = 2;
  cpu_pool = new_<thread_pool_t>(CPU,cpu_threads,0);
  if (io_threads)
    io_pool = new_<thread_pool_t>(IO,io_threads,1000);
  time_info.total_start = time_info.local_start = time();
}

void threads_schedule(thread_type_t type, const function<void()>& f) {
  OTHER_ASSERT(type==CPU || type==IO);
  if (type!=CPU) OTHER_ASSERT(io_pool);
  (type==CPU?cpu_pool:io_pool)->schedule(f);
}

void threads_schedule(thread_type_t type, const vector<function<void()>>& fs) {
  OTHER_ASSERT(type==CPU || type==IO);
  if (type!=CPU) OTHER_ASSERT(io_pool);
  (type==CPU?cpu_pool:io_pool)->schedule(fs);
}

void threads_wait_all() {
  if (!io_pool)
    cpu_pool->wait();
  else
    for (;;) {
      cpu_pool->wait();
      io_pool->wait();
      lock_t cpu_lock(cpu_pool->mutex);
      lock_t io_lock(io_pool->mutex);
      if (cpu_pool->error)
        cpu_pool->error.throw_();
      if (io_pool->error)
        io_pool->error.throw_();
      OTHER_ASSERT(!cpu_pool->die);
      OTHER_ASSERT(!io_pool->die);
      if (   !cpu_pool->jobs.size() && cpu_pool->waiting==cpu_pool->count
          && !io_pool->jobs.size() && io_pool->waiting==io_pool->count)
        break;
    }
}

// TODO: This function doesn't account for the case when the pool temporarily
// runs out of jobs, but currently running jobs schedule a bunch more.  If
// that happens, it safely but stupidly reverts to wait().
void threads_wait_all_help() {
  auto& pool = *cpu_pool;
  for (;;) {
    // Grab a job
    function<void()> f;
    {
      lock_t lock(pool.mutex);
      if (pool.die || !pool.jobs.size())
        goto wait;
      else {
        swap(f,pool.jobs.front());
        pool.jobs.pop_front();
      }
    }

    // Run the job
    try {
      f();
    } catch (const exception& e) {
      if (throw_callback)
        throw_callback(e.what());
      lock_t lock(pool.mutex);
      if (!pool.error)
        pool.error = e;
      pool.die = true;
      goto wait;
    }
  }
  wait:
  threads_wait_all();
}

/****************** time reports *****************/

void clear_thread_times() {
  OTHER_ASSERT(is_master());
  threads_wait_all();
  lock_t lock(time_mutex);
  double now = time();
  for (auto table : time_tables)
    for (HashtableIterator<const void*,time_entry_t> it(table->times);it.valid();it.next()) {
      time_entry_t& entry = it.data();
      if (entry.start) {
        if (history)
          entry.history.append(vec(entry.start,now));
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
  threads_wait_all();
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
    for (HashtableIterator<const void*,time_entry_t> it(table->times);it.valid();it.next()) {
      const string name = (const char*)it.key();
      time_entry_t& entry = it.data();
      if (total)
        type_to_name[table->type].set(name);
      if (entry.start) {
        if (history)
          entry.history.append(vec(entry.start,now));
        entry.local += now-entry.start;
        entry.start = now;
      }
      auto& time = times.get_or_insert(name,tuple(10000,0.));
      time.x = min(time.x,entry.id);
      entry.total += entry.local;
      double t = total?entry.total:entry.local;
      entry.local = 0;
      time.y += t;
      totals[table->type] += t;
    }

  // Sort by id
  set<Tuple<int,string>> order;
  for (HashtableIterator<string,Tuple<int,double>> it(times);it.valid();it.next())
    order.insert(tuple(it.data().x,it.key()));

  // Print times
  cout << "timing\n";
  for (auto& iname : order)
    cout << format("  %-20s %10.4f s\n",sanitize_name(iname.y.c_str()),times.get(iname.y.c_str()).y);
  if (io_pool)
    cout << format("  missing: master %.4f, cpu %.4f, io %.4f\n",elapsed-totals[MASTER],cpu_pool->count*elapsed-totals[CPU],io_pool->count*elapsed-totals[IO]);
  else
    cout << format("  missing: master %.4f, cpu %.4f\n",elapsed-totals[MASTER],cpu_pool->count*elapsed-totals[CPU]);
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

vector<Tuple<int,vector<Tuple<string,Array<Vector<double,2>>>>>> thread_history() {
  clear_thread_times();
  lock_t lock(time_mutex);
  vector<Tuple<int,vector<Tuple<string,Array<Vector<double,2>>>>>> data(time_tables.size());
  if (history)
    for (int t=0;t<(int)data.size();t++) {
      auto& table = *time_tables[t];
      data[t].x = table.thread_id;
      for (HashtableIterator<const void*,time_entry_t> it(table.times);it.valid();it.next())
        data[t].y.push_back(tuple(string((const char*)it.key()),it.data().history));
    }
  return data;
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
    threads_wait_all();

    // Compare
    OTHER_ASSERT(serial==parallel);
  }
}

}
using namespace pentago;

void wrap_thread() {
  OTHER_FUNCTION(set_thread_history)
  OTHER_FUNCTION(init_threads)
  OTHER_FUNCTION(thread_pool_test)
  OTHER_FUNCTION(clear_thread_times)
  OTHER_FUNCTION(report_thread_times)
  OTHER_FUNCTION(thread_history)
}
