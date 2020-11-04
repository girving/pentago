// Midgame solver in Metal

import Metal
import QuartzCore

func choose(_ n: Int, _ k: Int) -> Int {
  assert(n >= 0)
  let s = min(k, n - k)
  if s < 0 { return 0 }
  // c = choose(n, s) = choose(n-1, s-1) * n/s  = ... = (n-s-1)/1 * ... * n/s
  var c = 1
  if s != 0 {
    for i in 1...s {
      c = c * (n - s + i) / i
    }
  }
  return c
}

typealias Twenty<T> = (T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T)
typealias Offsets = Twenty<Int32>

func asarray<T>(_ o: Twenty<T>) -> [T] {
  [o.0, o.1, o.2, o.3, o.4, o.5, o.6, o.7, o.8, o.9, o.10, o.11, o.12, o.13, o.14, o.15, o.16, o.17, o.18, o.19]
}

class MidSolver {
  let device: MTLDevice
  let queue: MTLCommandQueue
  let lib: MTLLibrary
  
  // Compute kernels
  let sets1p: Compute
  let wins: Compute
  let cs1ps: Compute
  let set0Info: Compute
  let inner: Compute

  init() {
    device = MTLCreateSystemDefaultDevice()!
    queue = device.makeCommandQueue()!
    lib = device.makeDefaultLibrary()!
    sets1p = Compute(device, lib, "sets1p")
    wins = Compute(device, lib, "wins")
    cs1ps = Compute(device, lib, "cs1ps")
    set0Info = Compute(device, lib, "set0_info")
    inner = Compute(device, lib, "inner")
  }

  func solve(_ board: Board) -> [Board: Int] {
    let spots = 36 - board.count
    let workspace_size = board_workspace_size(board.s)
    let I = make_info_t(board.s, workspace_size)

    // Temporary buffers
    func total(_ offsets: Offsets) -> Int {
      Int(asarray(offsets)[spots + 1])
    }
    let workspace = BigGPUBuffer<halfsupers_t>(device, chunks: 4, count: Int(workspace_size))
    let sets1p = GPUBuffer<UInt64>(device, total(I.sets1p_offsets))
    let allWins = GPUBuffer<halfsuper_s>(device, total(I.wins_offsets))
    let cs1ps = GPUBuffer<UInt16>(device, total(I.cs1ps_offsets))
    let I0 = GPUBuffer<set0_info_t>(device, total(I.sets0_offsets))
    let results = Buffer<halfsupers_t>(device, 37 - board.count)

    // Compute pass
    let start = CACurrentMediaTime()
    let capture = makeCapture(device, on: false)
    let commands = queue.makeCommandBuffer()!
    let compute = commands.makeComputeCommandEncoder()!
    self.sets1p.run(compute, [Small(I), sets1p], sets1p.count)
    wins.run(compute, [Small(I), allWins], allWins.count)
    self.cs1ps.run(compute, [Small(I), sets1p, cs1ps], cs1ps.count)
    set0Info.run(compute, [Small(I), I0], I0.count)
    for n in (0...spots).reversed() {
      let N = make_inner_t(I, Int32(n))
      func slice<T>(_ buffer: GPUBuffer<T>, _ offsets: Offsets) -> GPUBuffer<T> {
        let o = asarray(offsets)
        return buffer[Int(o[n])..<Int(o[n+1])]
      }
      inner.run(compute, [Small(N),
                          slice(cs1ps, I.cs1ps_offsets),
                          slice(sets1p, I.sets1p_offsets),
                          slice(allWins, I.wins_offsets),
                          slice(I0, I.sets0_offsets), results] + workspace.chunks, Int(N.output.size))
    }
    compute.endEncoding()
    commands.commit()
    capture.stopCapture()

    commands.waitUntilCompleted()
    let elapsed = CACurrentMediaTime() - start
    print("midsolve: board \(board.name), slice \(board.count), time \(elapsed) s")

    // Gather results
    let supers = results.array()
    var values = Array(repeating: high_board_value_t(), count: 1+18+8*18)
    var count = 0
    supers.withUnsafeBufferPointer { s in
      values.withUnsafeMutableBufferPointer { v in
        count = Int(board_midsolve_traverse(board.s, s.baseAddress, v.baseAddress))
      }
    }
    let pairs = values.prefix(count).map { kv in return (Board(kv.board), Int(kv.value)) }
    return Dictionary(pairs, uniquingKeysWith: { (v0, v1) in assert(v0 == v1); return v0 })
  }
}
