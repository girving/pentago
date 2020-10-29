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
    let results = Buffer<mid_super_t>(device, 37 - board.count)

    // Start capture
    /*
    let capture = MTLCaptureManager.shared()
    let desc = MTLCaptureDescriptor()
    desc.captureObject = device
    try! capture.startCapture(with: desc)
    */

    // Compute pass
    let start = CACurrentMediaTime()
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
    //capture.stopCapture()
    
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
