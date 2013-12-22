// Node.js bindings for a tiny bit of pentago

#include <node.h>
#include <v8.h>
#include "board.h"
using namespace v8;
using v8::Object;

void init_all(Handle<v8::Object> exports) {
  js_high_board_t::init(exports);
}

NODE_MODULE(pentago,init_all)
