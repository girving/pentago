// Wrappers around MPI reductions to isolate MPI dependencies
#pragma once

#include <other/core/array/RawArray.h>
namespace pentago {
namespace end {

enum op_t { sum_op, max_op };

// Perform a reduction and return true if we're the root.  After the
// reduction, only the root has valid data.  Null indicates no reduction.
template<class T,op_t op> struct reduction_t : public function<bool(RawArray<T>)> {
  reduction_t() {}
  template<class F> explicit reduction_t(const F& f) : function<bool(RawArray<T>)>(f) {}
};

}
}
