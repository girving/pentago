// Specialize version of function<void()> for use in thread pools
#pragma once

// boost::function has the following disadvantages:
//
// 1. It is complicated, which in particular makes stack traces sad looking.
// 2. functor_manager appears to take a nontrivial amount of time.
//
// In hindsight, it looks like the functor_manager might have been justified
// in taking time since in one case it was deallocating a bunch of memory.
// However, I still prefer my version.

#include <other/core/utility/debug.h>
#include <other/core/utility/SanitizeFunction.h>
#include <other/core/utility/unique.h>
#include <boost/noncopyable.hpp>
namespace pentago {

using namespace other;

struct job_base_t : public boost::noncopyable {
  virtual ~job_base_t() {}
  virtual void operator()() const = 0;
};

namespace {
template<class F> struct job_helper_t : public job_base_t {
  const typename SanitizeFunction<F>::type f;

  template<class F_> job_helper_t(F_&& f)
    : f(f) {}

  virtual void operator()() const {
    f();
  }
};
}

struct job_t : public Unique<job_base_t> {
  job_t() {}

  template<class F> job_t(F&& f)
    : Unique<job_base_t>(new job_helper_t<F>(other::move(f))) {}

  void operator()() const {
    OTHER_ASSERT(get());
    (*get())();
  }
};

}
namespace std {
static inline void swap(pentago::job_t& f, pentago::job_t& g) {
  f.swap(g);
}
}
