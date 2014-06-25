// Aligned array allocation

#include <pentago/utility/aligned.h>
#include <pentago/utility/char_view.h>
#include <pentago/utility/debug.h>
#include <pentago/utility/memory.h>
#include <geode/array/Array2d.h>
#include <geode/python/wrap.h>
#include <vector>
namespace pentago {

using std::vector;
using std::bad_alloc;

namespace {

struct aligned_buffer_t : public Noncopyable {
  GEODE_DECLARE_TYPE(GEODE_NO_EXPORT) // Declare pytype
  GEODE_PY_OBJECT_HEAD // Reference counter and pointer to type object
  size_t size; // Size of memory block
  void* start; // Owning pointer to start of block
};

static void free_buffer(PyObject* object) {
  aligned_buffer_t* buffer = (aligned_buffer_t*)object;
  free(buffer->start);
  ssize_t size = buffer->size;
  free(buffer);
  report_large_alloc(-size);
}

#ifdef GEODE_PYTHON

PyTypeObject aligned_buffer_t::pytype = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                          // ob_size
  "pentago.aligned_buffer_t", // tp_name
  sizeof(aligned_buffer_t),   // tp_basicsize
  0,                          // tp_itemsize
  free_buffer,                // tp_dealloc
  0,                          // tp_print
  0,                          // tp_getattr
  0,                          // tp_setattr
  0,                          // tp_compare
  0,                          // tp_repr
  0,                          // tp_as_number
  0,                          // tp_as_sequence
  0,                          // tp_as_mapping
  0,                          // tp_hash 
  0,                          // tp_call
  0,                          // tp_str
  0,                          // tp_getattro
  0,                          // tp_setattro
  0,                          // tp_as_buffer
  Py_TPFLAGS_DEFAULT,         // tp_flags
  "Aligned memory buffer",    // tp_doc
  0,                          // tp_traverse
  0,                          // tp_clear
  0,                          // tp_richcompare
  0,                          // tp_weaklistoffset
  0,                          // tp_iter
  0,                          // tp_iternext
  0,                          // tp_methods
  0,                          // tp_members
  0,                          // tp_getset
  0,                          // tp_base
  0,                          // tp_dict
  0,                          // tp_descr_get
  0,                          // tp_descr_set
  0,                          // tp_dictoffset
  0,                          // tp_init
  0,                          // tp_alloc
  0,                          // tp_new
  0,                          // tp_free
};

#else // non-python stub

PyTypeObject aligned_buffer_t::pytype = {
  "pentago.aligned_buffer_t", // tp_name
  free_buffer,                // tp_dealloc
};

#endif

// All necessary aligned_buffer_t::pytpe fields are filled in, so no PyType_Ready is needed

}

Tuple<void*,PyObject*> aligned_buffer_helper(size_t alignment, size_t size) {
  if (!size)
    return tuple((void*)0,(PyObject*)0);
#ifndef __APPLE__
#ifdef __bgq__
  alignment = max(alignment,size_t(32)); // See https://wiki.alcf.anl.gov/parts/index.php/Blue_Gene/Q#Allocating_Memory
#endif
  void* start;
  if (posix_memalign(&start,alignment,size))
    THROW(bad_alloc);
  void* pointer = start;
#else
  // Mac OS 10.7.4 has a buggy version of posix_memalign, so do our own alignment at the cost of one extra element
  void* start = malloc(size+alignment-1);
  if (!start)
    THROW(bad_alloc);
  size_t p = (size_t)start;
  p = (p+alignment-1)&~(alignment-1);
  void* pointer = (void*)p;
#endif
  auto* buffer = (aligned_buffer_t*)malloc(sizeof(aligned_buffer_t));
  if (!buffer)
    THROW(bad_alloc);
  (void)GEODE_PY_OBJECT_INIT(buffer,&buffer->pytype);
  buffer->size = size;
  buffer->start = start;
  report_large_alloc(size);
  return tuple(pointer,(PyObject*)buffer);
}

// Most useful when run under valgrind to test for leaks
static void aligned_test() {
  // Check empty buffer
  struct large_t { char data[64]; };
  aligned_buffer<large_t>(0);

  vector<Array<uint8_t>> buffers;
  for (int i=0;i<100;i++) {
    // Test 1D
    auto x = aligned_buffer<int>(10);
    x.zero();
    x[0] = 1;
    x.back() = 2;
    GEODE_ASSERT(x.sum()==3);
    buffers.push_back(char_view_own(x));

    // Test 2D
    auto y = aligned_buffer<float>(vec(4,5));
    y.flat.zero();
    y(0,0) = 1;
    y(0,4) = 2;
    y(3,0) = 3;
    y(3,4) = 4;
    GEODE_ASSERT(y.sum()==10);
    buffers.push_back(char_view_own(y.flat));
  }

  for (auto& x : buffers)
    GEODE_ASSERT((((long)x.data())&15)==0);
}

}
using namespace pentago;

void wrap_aligned() {
  GEODE_FUNCTION(aligned_test)
}
