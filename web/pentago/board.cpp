// Node.js bindings for high_board_t

#include "board.h"
#include <geode/python/Ptr.h>

using namespace v8;
using namespace geode;
using namespace pentago;

static Persistent<Function> constructor;

js_high_board_t::js_high_board_t(const high_board_t& board)
  : board(ref(board)) {}

js_high_board_t::~js_high_board_t() {}

void js_high_board_t::init(Handle<v8::Object> exports) {
  // Prepare constructor template
  auto t = FunctionTemplate::New(new_);
  t->SetClassName(String::NewSymbol("high_board_t"));
  t->InstanceTemplate()->SetInternalFieldCount(1);

  // Add functions
  t->PrototypeTemplate()->Set(String::NewSymbol("name"),FunctionTemplate::New(name)->GetFunction());
  t->PrototypeTemplate()->Set(String::NewSymbol("done"),FunctionTemplate::New(done)->GetFunction());
  t->PrototypeTemplate()->Set(String::NewSymbol("moves"),FunctionTemplate::New(moves)->GetFunction());

  // Finish
  constructor = Persistent<Function>::New(t->GetFunction());
  exports->Set(String::NewSymbol("high_board_t"),constructor);
}

Handle<Value> js_high_board_t::new_(const Arguments& args) {
  HandleScope scope;
  if (args.IsConstructCall()) {
    // Invoked as constructor
    Ptr<high_board_t> board;
    try {
      if (args.Length()) {
        const String::AsciiValue name(args[0]);
        board = high_board_t::parse(*name);
      } else
        board = geode::new_<high_board_t>(0,false);
    } catch (const exception& e) {
      ThrowException(Exception::Error(String::New(e.what())));
      return scope.Close(Undefined());
    }
    auto self = new js_high_board_t(*board);
    self->Wrap(args.This());
    return args.This();
  } else {
    // Invoked as plain function, turn into construct call
    Local<Value> argv[1] = { args[0] };
    return scope.Close(constructor->NewInstance(1,argv));
  }
}

Handle<Value> js_high_board_t::name(const Arguments& args) {
  HandleScope scope;
  try {
    const auto self = ObjectWrap::Unwrap<js_high_board_t>(args.This());
    return scope.Close(String::New(str(*self->board).c_str()));
  } catch (const exception& e) {
    ThrowException(Exception::Error(String::New(e.what())));
    return scope.Close(Undefined());
  }
}

Handle<Value> js_high_board_t::done(const Arguments& args) {
  HandleScope scope;
  try {
    const auto self = ObjectWrap::Unwrap<js_high_board_t>(args.This());
    return scope.Close(Boolean::New(self->board->done()));
  } catch (const exception& e) {
    ThrowException(Exception::Error(String::New(e.what())));
    return scope.Close(Undefined());
  }
}

Handle<Value> js_high_board_t::moves(const Arguments& args) {
  HandleScope scope;
  try {
    const auto self = ObjectWrap::Unwrap<js_high_board_t>(args.This());
    auto moves = v8::Array::New();
    int n = 0;
    for (auto move : self->board->moves()) {
      // I don't know how to call a nonexposed constructor, so build a default constructed one and tweak
      auto js_move = constructor->NewInstance(0,0);
      const_cast_(ObjectWrap::Unwrap<js_high_board_t>(js_move)->board) = move;
      moves->Set(n++,js_move);
    }
    return scope.Close(moves);
  } catch (const exception& e) {
    ThrowException(Exception::Error(String::New(e.what())));
    return scope.Close(Undefined());
  }
}
