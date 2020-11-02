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

class MidSolver {
  let device: MTLDevice
  let queue: MTLCommandQueue
  let lib: MTLLibrary
  
  // Compute kernels
  let subsets: Compute
  let wins: Compute
  let cs1ps: Compute
  let set0Info: Compute
  let inner: Compute

  init() {
    device = MTLCreateSystemDefaultDevice()!
    queue = device.makeCommandQueue()!
    lib = device.makeDefaultLibrary()!
    subsets = Compute(device, lib, "subsets")
    wins = Compute(device, lib, "wins")
    cs1ps = Compute(device, lib, "cs1ps")
    set0Info = Compute(device, lib, "set0_info")
    inner = Compute(device, lib, "inner")
  }

  func solve(_ board: Board) -> [Board: Int] {
    let spots = 36 - board.count
    let maxSets = choose(spots, spots / 2)

    // Temporary buffers
    let workspace = BigGPUBuffer<halfsupers_t>(device, chunks: 4, count: Int(board_workspace_size(board.s)))
    let sets1p = GPUBuffer<UInt64>(device, maxSets)
    let allWins = GPUBuffer<halfsuper_s>(device, 2 * maxSets)
    let cs1ps = GPUBuffer<UInt16>(device, maxSets * spots)
    let I0 = GPUBuffer<set0_info_t>(device, maxSets)
    let results = Buffer<halfsupers_t>(device, 37 - board.count)

    // Sizes
    if false {
      var subsetsTotal = 0
      var winsTotal = 0
      var cs1psTotal = 0
      var set0InfoTotal = 0
      var set0InfoSizes: [String] = []
      var offset0Size = 0
      for n in (0...spots).reversed() {
        let I = make_info_t(board.s, Int32(n), Int32(workspace.count))
        let W = make_wins_info_t(I)
        subsetsTotal += Int(I.sets1p.size)
        winsTotal += Int(W.size)
        cs1psTotal += Int(I.cs1ps_size)
        set0InfoTotal += Int(I.sets0.size)
        set0InfoSizes.append(large(Int(I.sets0.size) * MemoryLayout<set0_info_t>.size))
        offset0Size = max(offset0Size, Int(I.k1 * (I.spots - I.k0)))
      }
      print("slice \(board.count) tmp buffers:")
      print("  subsets size = \(large(subsetsTotal * MemoryLayout<UInt64>.size))")
      print("  wins size = \(large(winsTotal * MemoryLayout<halfsuper_s>.size))")
      print("  cs1ps size = \(large(cs1psTotal * MemoryLayout<UInt16>.size))")
      print("  I0 size = \(large(set0InfoTotal * MemoryLayout<set0_info_t>.size))")
      print("  I0 sizes = \(set0InfoSizes)")
      print("  workspace size = \(large(workspace.count * MemoryLayout<halfsupers_t>.size))")
      print("  offset0 size = \(offset0Size)")
    }

    // Compute pass
    let start = CACurrentMediaTime()
    let capture = makeCapture(device, on: false)
    let commands = queue.makeCommandBuffer()!
    let compute = commands.makeComputeCommandEncoder()!
    for n in (0...spots).reversed() {
      let I = make_info_t(board.s, Int32(n), Int32(workspace.count))
      let W = make_wins_info_t(I)
      subsets.run(compute, [Small(I.sets1p), sets1p], Int(I.sets1p.size))
      wins.run(compute, [Small(W), allWins], Int(W.size))
      self.cs1ps.run(compute, [Small(I), sets1p, cs1ps], Int(I.cs1ps_size))
      set0Info.run(compute, [Small(I), allWins, I0], Int(I.sets0.size))
      inner.run(compute, [Small(I), cs1ps, sets1p, allWins, I0, results] + workspace.chunks, Int(I.sets0.size * I.sets1p.size))
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


/*
class Bounce: AnimationDelegate {
  let pipeline: MTLRenderPipelineState
  let stepPipeline: MTLComputePipelineState
  let motion: CMMotionManager
  let particles: Particles<BounceParticle>
  
  func draw(_ dt: Float) {
    let commands = anim.commands()
    let compute = commands.makeComputeCommandEncoder()!
    compute.setComputePipelineState(stepPipeline)
    compute.setBuffer(particles.data, offset: 0, index: 0)
    setBytes(compute, stepInfo(dt), index: 1)
    compute.setTexture(screenLevelset, index: 0)
    dispatch(compute, stepPipeline, s.count)
    compute.endEncoding()
    commands.commit()
  }
}

*/
