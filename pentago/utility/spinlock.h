// Spin locks
#pragma once

#include "pentago/utility/noncopyable.h"
#include <type_traits>
#ifdef __APPLE__
#include <os/lock.h>
#else
#include <pthread.h>
#endif
namespace pentago {

/* Important non-portability note:
 *
 * The whole point of a spin lock is to be cheap and compact.  Thus, a spin lock is probably
 * an int, and the idea of a spin lock having a nontrivial destructor seems silly.  Therefore,
 * our spinlock_t will have a trivial destructor.  If we use pthreads for spinlocks and the
 * destructor does something complicated, we're going to leak a bunch of spinlocks.  I can't
 * imagine this actually happening, but just to be safe I'm going to require GLIBC for now in
 * the pthreads case.
 */

// Apple has their own special version, which I actually like more than the pthreads version.
// In particular, they guarantee that a spinlock is an integer, and zero means unlocked.
#if defined(__APPLE__)

struct spinlock_t {
  os_unfair_lock spinlock;

  spinlock_t()
    : spinlock(OS_UNFAIR_LOCK_INIT) {}

  void lock() {
    os_unfair_lock_lock(&spinlock);
  }

  bool trylock() {
    return os_unfair_lock_trylock(&spinlock);
  }

  void unlock() {
    os_unfair_lock_unlock(&spinlock);
  }
};

#elif defined(__wasm__)

// Single-threaded, so no locks
struct spinlock_t {
  void lock() {}
  bool trylock() { return true; }
  void unlock() {}
};

// Use pthreads if we have them.
#elif defined(_POSIX_SPIN_LOCKS) && _POSIX_SPIN_LOCKS>0

// Warning: We explicitly break the pthread standard in two ways.
//
// 1. We assume pthread_spin_destroy is a no-op.
// 2. We ignore all error codes.
//
// I've checked that both are safe to do with glibc, so we restrict to that for now.
#ifdef __GLIBC__
static_assert(std::is_same<volatile int,pthread_spinlock_t>::value,"");
#else
#error "I'm not sure if pthread_spin_destroy is a no-op on this system"
#endif

struct spinlock_t {
  pthread_spinlock_t spinlock;

  spinlock_t() {
    // We use MPI for interprocess communication, so private is enough
    pthread_spin_init(&spinlock,PTHREAD_PROCESS_PRIVATE);
  }

  // Explicitly don't use a copy constructor.

  void lock() {
    pthread_spin_lock(&spinlock);
  }

  bool trylock() {
    return !pthread_spin_trylock(&spinlock);
  }

  void unlock() {
    pthread_spin_unlock(&spinlock);
  }
};

#else
#error "No available spin lock implementation"
#endif

// Convenience class to lock and unlock a spinlock
struct spin_t : noncopyable_t {
  spinlock_t& lock;

  spin_t(spinlock_t& lock)
    : lock(lock) {
    lock.lock();
  }

  ~spin_t() {
    lock.unlock();
  }
};

}
