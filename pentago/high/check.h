// Check high level interface against forward search
#pragma once

#include "pentago/high/board.h"
#include "pentago/data/block_cache.h"
NAMESPACE_PENTAGO

// 1 if the player to move wins, 0 for tie, -1 if the player to move loses
int value(const block_cache_t& cache, const high_board_t board);

// Same as value, but verify consistency with minimum depth tree search.
int value_check(const block_cache_t& cache, const high_board_t board);

// Compare against a bunch of samples and return loss,tie,win counts
Vector<int,3> sample_check(const block_cache_t& cache, RawArray<const board_t> boards,
                           RawArray<const Vector<super_t,2>> wins);

END_NAMESPACE_PENTAGO
