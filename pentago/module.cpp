// Pentago python module

#include <pentago/end/config.h>
#include <geode/python/module.h>
#include <geode/python/wrap.h>

GEODE_PYTHON_MODULE(pentago_core) {
  GEODE_WRAP(board)
  GEODE_WRAP(score)
  GEODE_WRAP(moves)
  GEODE_WRAP(stat)
  GEODE_WRAP(hash)
  GEODE_WRAP(table)
  GEODE_WRAP(engine)
  GEODE_WRAP(superengine)
  GEODE_WRAP(superscore)
  GEODE_WRAP(supertable)
  GEODE_WRAP(symmetry)
  GEODE_WRAP(all_boards)
  GEODE_WRAP(count)
  GEODE_WRAP(trace)
  GEODE_WRAP(file)
  GEODE_WRAP(lru)
  GEODE_WRAP(supertensor)
  GEODE_WRAP(filter)
  GEODE_WRAP(analyze)
  GEODE_WRAP(thread)
  GEODE_WRAP(aligned)
  GEODE_WRAP(section)
  GEODE_WRAP(block_cache)

  // Endgame
  GEODE_WRAP(sections)
  GEODE_WRAP(partition)
  GEODE_WRAP(simple_partition)
  GEODE_WRAP(random_partition)
  GEODE_WRAP(block_store)
  GEODE_WRAP(predict)
  GEODE_WRAP(check)
  GEODE_WRAP(history)
  GEODE_WRAP(fast_compress)
  GEODE_WRAP(load_balance)
  GEODE_WRAP(store_block_cache)
  GEODE_WRAP(compacting_store)
  GEODE_WRAP(verify)

  // Configuration
  GEODE_OBJECT_2(pentago_end_pad_io,pentago::end::pad_io)

  // High
  GEODE_WRAP(high_board)
  GEODE_WRAP(index)

  // Mid
  GEODE_WRAP(halfsuper)
  GEODE_WRAP(midengine)
}
