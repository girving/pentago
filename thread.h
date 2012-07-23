// Thread utilities
#pragma once

#include <other/core/python/Ptr.h>
#include <other/core/python/Object.h>
#include <other/core/python/ExceptionValue.h>
#include <boost/noncopyable.hpp>
#include <boost/function.hpp>
#include <vector>
#include <pthread.h>
namespace pentago {

using namespace other;
using std::vector;
using boost::function;
struct time_entry_t;

struct mutex_t : public boost::noncopyable {
  pthread_mutex_t mutex;

  mutex_t() {
    OTHER_ASSERT(!pthread_mutex_init(&mutex,0));
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

struct cond_t : public boost::noncopyable {
  mutex_t& mutex;
  pthread_cond_t cond;

  cond_t(mutex_t& mutex)
    : mutex(mutex) {
    OTHER_ASSERT(!pthread_cond_init(&cond,0));
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

class thread_time_t : public boost::noncopyable {
  time_entry_t& entry;
public:
  thread_time_t(const char* name);
  ~thread_time_t();
};

void clear_thread_times();
void report_thread_times(bool total);

enum thread_type_t { MASTER=0, CPU=1, IO=2 };
thread_type_t thread_type();

// Initialize thread pools
void init_threads(int cpu_threads, int io_threads);

// Schedule a job
void threads_schedule(thread_type_t type, const function<void()>& f);

// Schedule many jobs
void threads_schedule(thread_type_t type, const vector<function<void()>>& fs);

// Wait for all jobs to complete
void threads_wait_all();

}
