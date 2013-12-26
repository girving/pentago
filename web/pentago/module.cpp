// Node.js bindings for a tiny bit of pentago

#include "wrap.h"
#include <pentago/data/async_block_cache.h>
#include <pentago/end/sections.h>
#include <pentago/high/board.h>
#include <pentago/high/index.h>
#include <pentago/search/supertable.h>
namespace pentago {
namespace node {

using namespace pentago::end;

static Ref<high_board_t> make_board(const Arguments& args) {
  return args.Length() ? high_board_t::parse(*String::AsciiValue(args[0])) : new_<high_board_t>(0,false);
}

static Ref<sections_t> make_sections(const Arguments& args) {
  throw TypeError("sections_t can only be made via descendent_sections()");
}

static Ref<block_cache_t> make_cache(const Arguments& args) {
  throw TypeError("block_cache_t is an abstract class, and cannot be constructed directly");
}

static Ref<async_block_cache_t> make_async(const Arguments& args) {
  if (args.Length() < 1 || !args[0]->IsNumber())
    throw TypeError("async_block_cache_t: expected one numeric memory_limit argument");
  return new_<async_block_cache_t>(uint64_t(args[0]->NumberValue()));
}

static Ref<supertensor_index_t> make_index(const Arguments& args) {
  if (args.Length() < 1)
    throw TypeError(format("supertensor_index_t: expected one sections_t argument, got %d arguments",args.Length()));
  return new_<supertensor_index_t>(from_js<const sections_t&>(args[0]));
}

static void init(Handle<v8::Object> exports) {
  {
    PN_CLASS(high_board_t,make_board)
    PN_METHOD(name)
    PN_METHOD(done)
    PN_METHOD(count)
    PN_METHOD(turn)
    PN_METHOD(middle)
    PN_METHOD(moves)
    PN_METHOD(value)
    PN_METHOD(immediate_value)
  } {
    PN_CLASS(sections_t,make_sections)
  } {
    PN_CLASS(block_cache_t,make_cache)
  } {
    PN_CLASS(async_block_cache_t,make_async)
    PN_METHOD(board_block)
    PN_METHOD(contains)
    PN_METHOD(set)
  } {
    PN_CLASS(supertensor_index_t,make_index)
    PN_METHOD(blob_location)
    PN_METHOD(block_location)
  }
  PN_FUNCTION(descendent_sections)
  PN_FUNCTION(init_supertable)
  PN_FUNCTION(empty_block_cache)
  PN_FUNCTION(init_threads)
}

}
}
NODE_MODULE(pentago,pentago::node::init)
