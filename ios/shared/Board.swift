// Board: Swift version of high_board_t

func ipow(_ a: Int, _ b: Int) -> Int {
  Int(pow(Double(a), Double(b)))
}

func pack(_ side0: UInt64, _ side1: UInt64) -> UInt64 {
  var board: UInt64 = 0
  for q in 0..<4 {
    var quad: UInt64 = 0
    for i in (0..<9).reversed() {
      let n = UInt64(16*q + i)
      let x = ((side0 >> n) & 1) + 2*((side1 >> n) & 1)
      quad = 3*quad + x
    }
    board += quad << UInt64(16*q)
  }
  return board
}

func unpack(_ board: UInt64) throws -> (UInt64, UInt64) {
  var side0: UInt64 = 0
  var side1: UInt64 = 0
  for q in 0..<4 {
    let quad = Int((board >> UInt64(16*q)) & 0xffff)
    if quad > ipow(3, 9) {
      throw DecodingError.dataCorrupted(
        DecodingError.Context(codingPath: [], debugDescription: "\(board) has invalid quadrant \(q)"))
    }
    for i in 0..<9 {
      let x = quad / ipow(3, i) % 3
      let bit = UInt64(1) << UInt64(16*q+i)
      if x == 1 { side0 += bit }
      if x == 2 { side1 += bit }
    }
  }
  return (side0, side1)
}

enum Stone: Int {
  case empty = 0
  case black = 1
  case white = 2
}

struct Board: Hashable, CustomStringConvertible {
  let s: high_board_s

  init() {
    s = high_board_s(side_: (0,0), ply_: 0)
  }

  init(_ b: high_board_s) {
    s = b
  }

  init(_ name: String) throws {
    var ns = name
    let middle = name.hasSuffix("m")
    if middle {
      ns.removeLast()
    }
    if let n = UInt64(ns) {
      let (side0, side1) = try unpack(n)
      s = high_board_s(side_: (side0, side1), ply_: UInt32(2*(side0 | side1).nonzeroBitCount - (middle ? 1 : 0)))
    } else {
      throw DecodingError.dataCorrupted(
        DecodingError.Context(codingPath: [], debugDescription: "Expected \\d+m?, got '\(name)'"))
    }
  }

  static func ==(a: Board, b: Board) -> Bool {
    a.s.side_ == b.s.side_ && a.s.ply_ == b.s.ply_
  }

  func hash(into hasher: inout Hasher) {
    hasher.combine(s.side_.0)
    hasher.combine(s.side_.1)
    hasher.combine(s.ply_)
  }

  var name: String {
    let m = middle ? "m" : ""
    return "\(pack(s.side_.0, s.side_.1))\(m)"
  }
  var description: String { name }

  var count: Int { Int((s.ply_ + 1) >> 1) }
  var middle: Bool { s.ply_ & 1 != 0}
  var turn: Bool { (s.ply_ >> 1) & 1 != 0 }
  var turnStone: Stone { turn ? .white : .black }
  var emptyMask: UInt64 { 0x01ff01ff01ff01ff ^ s.side_.0 ^ s.side_.1 }

  func done() -> Bool { board_done(s) }
  func immediateValue() -> Int { Int(board_immediate_value(s)) }

  // Map quadrant and spot id to stone
  func stone(q: Int, k: Int) -> Stone {
    assert(0 <= q && q < 4 && 0 <= k && k < 9)
    func bit(_ n: UInt64) -> Bool { n >> UInt64(16*q+k) & 1 != 0 }
    return bit(s.side_.0) ? .black : bit(s.side_.1) ? .white : .empty
  }

  // Place a stone, if the move is valid
  func place(q: Int, k: Int) -> Board? {
    assert(0 <= q && q < 4 && 0 <= k && k < 9)
    return middle || stone(q: q, k: k) != .empty ? .none : Board(board_place_bit(s, Int32(16*q+k)))
  }

  // All valid moves
  func moves() -> [Board] {
    var moves: [Board] = []
    if !middle {  // Place a stone
      let empty = emptyMask
      for bit in 0..<64 {
        if (empty >> UInt64(bit) & 1) != 0 {
          moves.append(Board(board_place_bit(s, Int32(bit))))
        }
      }
    } else {  // Rotate a quadrant
      for q in 0..<4 {
        for d in [-1, 1] {
          moves.append(Board(board_rotate(s, Int32(q), Int32(d))))
        }
      }
    }
    return moves
  }

  func midsolve() -> [Board: Int] {
    assert(count >= 18, "\(name): count \(count)")
    var buffer = Array(repeating: high_board_value_t(), count: 1+18+8*18)
    var size = 0
    buffer.withUnsafeMutableBufferPointer { p in size = Int(board_midsolve(s, p.baseAddress)) }
    let pairs = buffer.prefix(size).map { kv in return (Board(kv.board), Int(kv.value)) }
    return Dictionary(pairs, uniquingKeysWith: { (v0, v1) in assert(v0 == v1); return v0 })
  }
}
