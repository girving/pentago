// Partially apply a function object.
// This is essentially a simpler version of boost::bind.
#pragma once

#include <type_traits>
#include <utility>
namespace pentago {

namespace {

template<class F> struct sanitize_function { typedef F type; };
template<class F> struct sanitize_function<F&> : public sanitize_function<F> {};
template<class F> struct sanitize_function<F&&> : public sanitize_function<F> {};

template<class T,class R,class... Args> struct sanitize_method {
  typedef std::remove_const_t<T> T_;
  typedef std::conditional_t<std::is_const<T>::value,R(T_::*)(Args...) const,R(T_::*)(Args...)> F;
  F f;

  sanitize_method(F f) : f(f) {}

  template<class... Rest> R operator()(T* self, Rest&&... args) const {
    return (self->*f)(std::forward<Rest>(args)...);
  }

  template<class... Rest> R operator()(T& self, Rest&&... args) const {
    return (self.*f)(std::forward<Rest>(args)...);
  }
};

template<class R,class... Args> struct sanitize_function<R(Args...)> {
  typedef R (*type)(Args...);
};

template<class T,class R,class... Args> struct sanitize_function<R(T::*)(Args...)> {
  typedef sanitize_method<T,R,Args...> type;
};

template<class T,class R,class... Args> struct sanitize_function<R(T::*)(Args...) const> {
  typedef sanitize_method<const T,R,Args...> type;
};

template<class F, class I, class Args> struct Curry;

template<class F, size_t... indices, class Args>
struct Curry<F, std::index_sequence<indices...>, Args> {
  const typename sanitize_function<F>::type f;
  const Args args;

  template<class... Rest> auto operator()(Rest&&... rest) const {
    return f(std::get<indices>(args)..., std::forward<Rest>(rest)...);
  }
};

template<class F, class G> struct Compose {
  const typename sanitize_function<F>::type f;
  const typename sanitize_function<G>::type g;

  template<class... Args> auto operator()(Args&&... args) const {
    return f(g(args...));
  }
};

}

template<class F, class... Args> static inline auto curry(F&& f, Args&&... args) {
  typedef std::tuple<std::remove_reference_t<Args>...> Tuple;
  return Curry<F, std::index_sequence_for<Args...>, Tuple>({
      std::forward<F>(f), Tuple(std::forward<Args>(args)...)});
}

template<class F,class G> static inline auto compose(F&& f, G&& g) {
  return Compose<F,G>({std::forward<F>(f), std::forward<G>(g)});
}

}  // namespace pentago
