// Array allocation using mmap

#include <pentago/utility/mmap.h>
#include <pentago/utility/char_view.h>
#include <pentago/utility/debug.h>
#include <pentago/utility/memory.h>
#include <geode/array/Array2d.h>
#include <geode/python/wrap.h>
#include <sys/mman.h>
#include <vector>
namespace pentago {

using std::vector;
using std::bad_alloc;

namespace {

struct mmap_buffer_t : public boost::noncopyable {
  GEODE_DECLARE_TYPE(GEODE_NO_EXPORT) // Declare pytype
  GEODE_PY_OBJECT_HEAD // Reference counter and pointer to type object
  size_t size; // Size of memory block
  void* start; // Owning pointer to start of block
};

static void free_buffer(PyObject* object) {
  mmap_buffer_t* buffer = (mmap_buffer_t*)object;
  munmap(buffer->start,buffer->size);
  ssize_t size = buffer->size;
  free(buffer);
  report_large_alloc(-size);
}

#ifdef GEODE_PYTHON

PyTypeObject mmap_buffer_t::pytype = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                          // ob_size
  "pentago.mmap_buffer_t",    // tp_name
  sizeof(mmap_buffer_t),      // tp_basicsize
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

PyTypeObject mmap_buffer_t::pytype = {
  "pentago.mmap_buffer_t",    // tp_name
  free_buffer,                // tp_dealloc
};

#endif

// All necessary mmap_buffer_t::pytpe fields are filled in, so no PyType_Ready is needed

}

Tuple<void*,PyObject*> mmap_buffer_helper(size_t size) {
  if (!size)
    return tuple((void*)0,(PyObject*)0);
  void* start = mmap(0,size,PROT_READ|PROT_WRITE,MAP_ANON|MAP_PRIVATE,-1,0);
  if (start==MAP_FAILED)
    THROW(RuntimeError,"anonymous mmap failed, size = %zu",size);
  auto* buffer = (mmap_buffer_t*)malloc(sizeof(mmap_buffer_t));
  if (!buffer)
    THROW(bad_alloc);
  (void)GEODE_PY_OBJECT_INIT(buffer,&buffer->pytype);
  buffer->size = size;
  buffer->start = start;
  report_large_alloc(size);
  return tuple(start,(PyObject*)buffer);
}

}
