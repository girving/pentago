// Pentago python module

#include <other/core/python/module.h>

OTHER_PYTHON_MODULE(pentago_core) {
  OTHER_WRAP(board)
  OTHER_WRAP(score)
  OTHER_WRAP(moves)
  OTHER_WRAP(stat)
  OTHER_WRAP(hash)
  OTHER_WRAP(table)
  OTHER_WRAP(engine)
  OTHER_WRAP(superengine)
  OTHER_WRAP(superscore)
  OTHER_WRAP(supertable)
  OTHER_WRAP(symmetry)
  OTHER_WRAP(all_boards)
  OTHER_WRAP(count)
  OTHER_WRAP(trace)
  OTHER_WRAP(supertensor)
  OTHER_WRAP(endgame)
  OTHER_WRAP(filter)
  OTHER_WRAP(analyze)
  OTHER_WRAP(thread)
  OTHER_WRAP(aligned)
  OTHER_WRAP(section)
  OTHER_WRAP(block_cache)

  // Endgame
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
  OTHER_WRAP(compacting_store)
}
