// Node.js bindings for high_board_t
#pragma once

#include <node.h>
#include <v8.h>
#include <pentago/high/board.h>
#include <geode/python/Ref.h>
using namespace v8;

class js_high_board_t : public node::ObjectWrap {
  const geode::Ref<const pentago::high_board_t> board;
public:
  static void init(Handle<v8::Object> exports);
private:
  js_high_board_t(const pentago::high_board_t& board);
  ~js_high_board_t();
  static Handle<Value> new_(const Arguments& args);
  static Handle<Value> name(const Arguments& args);
  static Handle<Value> done(const Arguments& args);
  static Handle<Value> moves(const Arguments& args);
};
