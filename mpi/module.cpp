// Pentago MPI python module

#include <other/core/python/module.h>

OTHER_PYTHON_MODULE(pentago_mpi) {
  OTHER_WRAP(sections)
  OTHER_WRAP(partition)
  OTHER_WRAP(simple_partition)
  OTHER_WRAP(random_partition)
  OTHER_WRAP(block_store)
  OTHER_WRAP(predict)
  OTHER_WRAP(check)
  OTHER_WRAP(history)
  OTHER_WRAP(fast_compress)
  OTHER_WRAP(load_balance)
  OTHER_WRAP(store_block_cache)
}
