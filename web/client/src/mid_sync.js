// Javascript interface to WebAssembly midsolver

// rollup will strip this in the browser case
import fs from 'fs'

// Compile mid.wasm
const code = fs.readFileSync ? Promise.resolve(fs.readFileSync('../public/mid.wasm'))
                             : fetch('/mid.wasm').then(r => r.arrayBuffer())
const mid_module = code.then(b => WebAssembly.compile(b))

// We inline str_to_quadrants and quadrants_to_str here to avoid having to include a module
// Since javascript doesn't support 64 bit ints, we have to jump through some hoops.
// Quadrants are little endian (0 through 3).  The follow tables are generated by 'pentago/web/generate quadrants'
const digit_quadrants = [[1,0,0,0],[10,0,0,0],[100,0,0,0],[1000,0,0,0],[10000,0,0,0],[34464,1,0,0],[16960,15,0,0],    [38528,152,0,0],[57600,1525,0,0],[51712,15258,0,0],[58368,21515,2,0],[59392,18550,23,0],[4096,54437,232,0],[40960,    20082,2328,0],[16384,4218,23283,0],[32768,42182,36222,3],[0,28609,34546,35],[0,23946,17784,355],[0,42852,46771,3552]]
const bit_sections = [[1,0,0,0],[2,0,0,0],[4,0,0,0],[8,0,0,0],[16,0,0,0],[32,0,0,0],[64,0,0,0],[128,0,0,0],[256,0,0,  0],[512,0,0,0],[1024,0,0,0],[2048,0,0,0],[4096,0,0,0],[8192,0,0,0],[16384,0,0,0],[32768,0,0,0],[65536,0,0,0],[31072,1,0,0],[62144,2,0,0],[24288,5,0,0],[48576,10,0,0],[97152,20,0,0],[94304,41,0,0],[88608,83,0,0],[77216,167,0,0],[54432,  335,0,0],[8864,671,0,0],[17728,1342,0,0],[35456,2684,0,0],[70912,5368,0,0],[41824,10737,0,0],[83648,21474,0,0],[67296,42949,0,0],[34592,85899,0,0],[69184,71798,1,0],[38368,43597,3,0],[76736,87194,6,0],[53472,74389,13,0],[6944,48779,27, 0],[13888,97558,54,0],[27776,95116,109,0],[55552,90232,219,0],[11104,80465,439,0],[22208,60930,879,0],[44416,21860,   1759,0],[88832,43720,3518,0],[77664,87441,7036,0],[55328,74883,14073,0],[10656,49767,28147,0],[21312,99534,56294,0],  [42624,99068,12589,1],[85248,98136,25179,2],[70496,96273,50359,4],[40992,92547,719,9],[81984,85094,1439,18],[63968,   70189,2879,36],[27936,40379,5759,72],[55872,80758,11518,144],[11744,61517,23037,288],[23488,23034,46075,576],[46976,  46068,92150,1152],[93952,92136,84300,2305],[87904,84273,68601,4611],[75808,68547,37203,9223]]

function str_to_quadrants(s) {
  // We assume s consists entirely of digits
  if (!s.match(/^\d+$/) || s.length>19)
    throw Error('expected number < 5540271966595842048, got '+s)
  // Accumulate into base 2**16, ignoring overflow
  const quads = [0,0,0,0]
  for (let i=0;i<s.length;i++) {
    const d = parseInt(s[s.length-1-i])
    for (let a=0;a<4;a++)
      quads[a] += d*digit_quadrants[i][a]
  }
  // Reduce down to base 2**16
  for (let i=0;i<3;i++) {
    quads[i+1] += quads[i]>>16
    quads[i] &= (1<<16)-1
  }
  if (quads[3]>=(1<<16))
    throw Error('expected number < 5540271966595842048, got '+s)
  return quads
}

function quadrants_to_str(quads) {
  // Accumulate into base 10**5, ignoring overflow
  const sections = [0,0,0,0]
  for (let b=0;b<64;b++)
    if (quads[b>>4]&1<<(b&15))
      for (let a=0;a<4;a++)
        sections[a] += bit_sections[b][a]
  // Reduce down to base 10**5
  let s = ''
  for (let i=0;i<3;i++) {
    sections[i+1] += Math.floor(sections[i]/100000)
    const si = '00000'+sections[i]%100000
    s = si.substr(si.length-5)+s
  }
  s = sections[3]+s
  return s.substr(Math.min(s.match(/^0*/)[0].length,s.length-1))
}

// Transform from ternary packed form to binary unpacked form, counting bits as we go
function unpack_quadrants(quads) {
  const s0 = [0, 0, 0, 0], s1 = [0, 0, 0, 0]
  let slice = 0
  for (let q = 0; q < 4; q++) {
    for (let i = 0; i < 9; i++) {
      const n = Math.floor(quads[q] / Math.pow(3, i)) % 3
      s0[q] |= (n == 1) << i
      s1[q] |= (n == 2) << i
      slice += n != 0
    }
  }
  return [s0, s1, slice]
}

function pack_quadrant(side0, side1) {
  let quad = 0
  for (let i = 0; i < 9; i++) {
    const mask = 1 << i
    quad += ((side0 >> i & 1) + 2 * (side1 >> i & 1)) * Math.pow(3, i)
  }
  return quad
}

// Parse into [...side0 quads, ...side1 quads, ply] format
function parse_board(name) {
  const m = name.match(/^(\d+)(m?)$/)
  if (!m)
    throw Error('Invalid board ' + name)
  const [side0, side1, slice] = unpack_quadrants(str_to_quadrants(m[1]))
  const middle = m[2].length
  return [...side0, ...side1, 2*slice-middle]
}

// Turn [...quads, middle] format into string
function show_board(board) {
  return quadrants_to_str(board.slice(0, 4)) + (board[4] ? 'm' : '')
}

function read_char_p(M, p) {
  const chars = new Uint8Array(M.exports.memory.buffer)
  let s = ''
  for (;;) {
    let c = chars[p++]
    if (!c) break
    s += String.fromCharCode(c)
  }
  return s
}

function read_int(M, p) {
  return new Int32Array(M.exports.memory.buffer, p, 1)[0]
}

// Read high_board_t into packed [...quads, middle] format
function read_board(M, p) {
  const high = new Uint16Array(M.exports.memory.buffer, p, 9)
  const quads = [0, 1, 2, 3].map(q => pack_quadrant(high[q], high[4+q]))
  const ply = high[8]
  return [...quads, ply & 1]
}

function write_board(M, p, board) {
  const sides = new Uint16Array(M.exports.memory.buffer, p, 8)
  for (let i = 0; i < 8; i++)
    sides[i] = board[i]
  const ply = new Uint32Array(M.exports.memory.buffer, p + 16, 1)
  ply[0] = board[8]
}

// Instantiate a fresh copy
export async function instantiate() {
  const module = await mid_module
  let M = null
  M = await WebAssembly.instantiate(module, {
    env: {
      die: msg_p => { throw Error(read_char_p(M, msg_p)) },
      wasm_log: (name_p, value) => console.log(read_char_p(M, name_p) + value),
    }
  })
  return M
}

// Raw interface to wasm_midsolve
export async function midsolve(board_name) {
  const M = await instantiate()

  // Allocate memory for arguments and results
  const limit = M.exports.midsolve_results_limit()
  const board_p = M.exports.wasm_malloc(24 + 8 + 32 * limit)
  const results_p = board_p + 24
  write_board(M, board_p, parse_board(board_name))

  // Compute!
  M.exports.wasm_midsolve(board_p, results_p)

  // Read results
  const results = {}
  const num_results = read_int(M, results_p)
  for (let i = 0; i < num_results; i++) {
    const tuple_p = results_p + 8 + 32 * i
    results[show_board(read_board(M, tuple_p))] = read_int(M, tuple_p + 24)
  }
  return results
}
