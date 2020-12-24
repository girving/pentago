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
  
  // Compute kernels (TODO: clean this list)
  let sets1p: Compute
  let wins1: Compute
  let cs1ps: Compute
  let set0Info: Compute
  let inner: Compute
  let set1Info: Compute
  let wins0: Compute
  let cs0ps: Compute
  let transposed: Compute

  init() {
    device = MTLCreateSystemDefaultDevice()!
    queue = device.makeCommandQueue()!
    lib = device.makeDefaultLibrary()!
    sets1p = Compute(device, lib, "sets1p")
    wins1 = Compute(device, lib, "wins1")
    cs1ps = Compute(device, lib, "cs1ps")
    set0Info = Compute(device, lib, "set0_info")
    inner = Compute(device, lib, "inner")
    set1Info = Compute(device, lib, "set1_info")
    wins0 = Compute(device, lib, "wins0")
    cs0ps = Compute(device, lib, "cs0ps")
    transposed = Compute(device, lib, "transposed")
  }

  func solve(_ board: Board) -> [Board: Int] {
    let start = CACurrentMediaTime()
    let spots = 36 - board.count
    let workspace_size = board_workspace_size(board.s)
    func total(_ offsets: Offsets) -> Int {
      Int(asarray(offsets)[spots + 1])
    }

    // Temporary buffers
    let I = make_transposed_t(board.s)
    let results = Buffer<halfsupers_t>(device, 37 - board.count)
    let workspace = BigGPUBuffer<halfsupers_t>(device, chunks: 4, count: Int(workspace_size))
    let I1s = GPUBuffer<set1_info_t>(device, total(I.sets1_offsets))
    let wins0 = GPUBuffer<halfsuper_s>(device, total(I.sets0_offsets))
    let cs0ps = GPUBuffer<UInt16>(device, total(I.cs0ps_offsets))

    // Prepare compute pass
    let capture = makeCapture(device, on: false, once: true)
    let commands = queue.makeCommandBuffer()!
    let compute = commands.makeComputeCommandEncoder()!

    // Helper computations
    self.set1Info.run(compute, [Small(I), I1s], I1s.count)
    self.wins0.run(compute, [Small(I), wins0], wins0.count)
    self.cs0ps.run(compute, [Small(I), cs0ps], cs0ps.count)

    // Midsolve!
    // We use manual binding to avoid rundandant PipelineStateObject binding warnings
    compute.setComputePipelineState(transposed.pipeline)
    set(compute, [I1s, wins0, cs0ps, results] + workspace.chunks, 1..<9)
    for n in (0...spots).reversed() {
      let N = make_transposed_inner_t(I, Int32(n))
      setBytes(compute, N, index: 0)
      dispatch(compute, transposed.pipeline, Int(N.grid.0) * Int(N.grid.1))
    }
    compute.endEncoding()
    commands.commit()
    capture.stop()
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
