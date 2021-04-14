// Asynchronous block cache
//
// A Javascript version of block_cache_t that does not know how to get its own
// data.  This is used to lift control flow into the asynchronous world of
// node.js, which is responsible to actually retrieving the data via
// asynchronous range requests.

'use strict'
const board_t = require('./board.js')
const tables = require('./tables.js')
const lru_t = require('lru-cache')
const lzma = require('lzma-native')
const ceil = Math.ceil
const floor = Math.floor
const max = Math.max
const min = Math.min
const pow = Math.pow

// Parameters
const block_size = 8

// Useful types
//   type section_t = [[int]*2]*4
//   type block_t = [section_t, [int]*4]

// Flakiness for unit tests
let flake_probability = 0
const set_flaky = p => { flake_probability = p }

function board_section(board) {  // → section_t
  const section = [[0,0],[0,0],[0,0],[0,0]]
  for (let q = 0; q < 4; q++) {
    const quad = board.quads[q]
    for (let i = 0; i < 9; i++) {
      const s = floor(quad / pow(3, i)) % 3
      if (s)
        section[q][s - 1]++
    }
  }
  return section
}

function transform_section(global, section) {  // → section_t
  const source = [[0,1,2,3],[1,3,0,2],[3,2,1,0],[2,0,3,1]][global & 3]
  const t = [section[source[0]], section[source[1]], section[source[2]], section[source[3]]]
  if (global & 4)
    [t[0], t[3]] = [t[3], t[0]]
  return t
}

function section_sig(s) {
  const n =  s[0][0] | s[0][1]<<4 | s[1][0]<<8 | s[1][1]<<12 | s[2][0]<<16 | s[2][1]<<20 | s[3][0]<<24 | s[3][1]<<28
  return n < 0 ? n + 4294967296 : n
}

function standardize_section(section) {  // → [section_t, global]
  let best = section
  let best_sig = section_sig(best)
  let best_g = 0
  for (let g = 0; g < 8; g++) {
    const t = transform_section(g, section)
    const t_sig = section_sig(t)
    if (best_sig > t_sig) {
      best = t
      best_sig = t_sig
      best_g = g
    }
  }
  return [best, best_g]
}

function section_sum(section) {  // → int
  return section.reduce((a, s) => a + s[0] + s[1], 0)
}

function section_shape(section) {  // → int
  const offsets = tables.rotation_minimal_quadrants_offsets
  return section.map(([black, white]) => {
    const i = ((black * (21-black)) >> 1) + white
    return offsets[i + 1] - offsets[i]
  })
}

function block_shape(section, block) {  // → int
  return section_shape(section).map((s, i) => min(block_size, s - block_size * block[i]))
}

// Transformation matrixes: 4 rotations followed by 4 reflections.  Generated by
//   def cs(a): return str(a) if a.ndim == 0 else '[%s]' % ','.join(map(cs, a))
//   mpow = np.linalg.matrix_power
//   rotate = np.array([[0,-1],[1,0]])
//   reflect = np.array([[0,-1],[-1,0]])
//   cs(np.asarray([mpow(reflect, n//4) @ mpow(rotate, n%4) for n in range(8)]))
const matrices = [
    [[1,0],[0,1]],[[0,-1],[1,0]],[[-1,0],[0,-1]],[[0,1],[-1,0]],
    [[0,-1],[-1,0]],[[-1,0],[0,1]],[[0,1],[1,0]],[[1,0],[0,-1]]
]

// Global board transformation, operating in ternary, returning quads rather than board_t
function global_transform_board(global, board) {  // → quads = [number]*4
  const place = (x, y) => pow(3, 3*(x%3)+(y%3))
  const M = matrices[global]
  const quads = [0, 0, 0, 0]
  for (let x = 0; x < 6; x++) {
    for (let y = 0; y < 6; y++) {
      const dx = (5 + M[0][0]*(2*x-5) + M[0][1]*(2*y-5)) / 2
      const dy = (5 + M[1][0]*(2*x-5) + M[1][1]*(2*y-5)) / 2
      const q = 2*floor(x/3) + floor(y/3)
      const dq = 2*floor(dx/3) + floor(dy/3)
      quads[dq] += place(dx, dy) * (floor(board.quads[q] / place(x, y)) % 3)
    }
  }
  return quads
}

// Uninterleave buffer viewed as Array<Vector<super_t,2>>
function uninterleave(data) {
  // Each 512 bit sequence of the form
  //   a0 b0 a1 b1 ... a255 b255
  // is turned into
  //   a0 a1 ... a255 b0 b1 ... b255
  if (data.length % 64)
    throw Error('uninterleave: length ' + data.length + ' not a multiple of 64')
  const out = Buffer.alloc(data.length)
  for (let i = 0; i < data.length; i += 64) {
    for (let j = 0; j < 32; j++) {
      let x = data[i + j * 2] | data[i + j * 2 + 1] << 8
      x = (x | x << 15) & 0x55555555
      x = (x | x >> 1) & 0x33333333
      x = (x | x >> 2) & 0x0f0f0f0f
      x = (x | x >> 4)
      out[i + j] = x & 0xff
      out[i + j + 32] = x >> 16 & 0xff
    }
  }
  return out
}

// Compute all sections that root depends on, organized by slice
function descendent_sections(max_slice) {  // → [[section_t]]
  const root = [[0,0],[0,0],[0,0],[0,0]]
  if (!(0 <= max_slice && max_slice <= 18))
    throw Error('invalid max_slice ' + max_slice)

  // Recursively compute all sections that root depends on
  const slices = []
  for (let s = 0; s <= max_slice; s++)
    slices.push([])
  const seen = new Set()
  const stack = [root]
  while (stack.length) {
    const s = standardize_section(stack.pop())[0]
    if (!seen.has('' + s)) {
      seen.add('' + s)
      const n = section_sum(s)
      slices[n].push(s)
      if (n < max_slice) {
        for (let q = 0; q < 4; q++) {
          if (s[q][0] + s[q][1] < 9) {
            const child = s.map(c => c.slice())
            child[q][n & 1]++
            stack.push(child)
          }
        }
      }
    }
  }

  // Sort each slice
  for (const slice of slices)
    slice.sort((a,b) => section_sig(a) - section_sig(b))
  return slices
}

function supertensor_index_t(sections) {
  const compact_blob_size = 12
  const section_blocks = s => section_shape(s).map(n => ceil(n / block_size))

  // Offsets into the index file
  let offset = 24
  const section_offset = {}
  for (const s of sections) {
    section_offset['' + s] = offset
    const blocks = section_blocks(s)
    offset += compact_blob_size * blocks[0] * blocks[1] * blocks[2] * blocks[3]
  }

  // Where is the compact_blob_t for this block in the index file?
  this.blob_location = block => {
    const [section, I] = block
    const shape = section_blocks(section)
    for (let i = 0; i < 4; i++)
      if (!(0 <= I[i] && I[i] < shape[i]))
        throw Error('section [' + section + '], shape [' + shape + '], invalid block [' + I + ']')
    return {offset: section_offset['' + section] +
                    compact_blob_size * ((((I[0] * shape[1]) + I[1]) * shape[2] + I[2]) * shape[3] + I[3]),
            size: compact_blob_size}
  }

  // Decode a compact_blob_t
  this.block_location = blob => {
    if (blob.length != compact_blob_size)
      throw Error('expected size ' + compact_blob_size + ', got size ' + blob.length)
    return {offset: blob.readUInt32LE(0) + blob.readUInt32LE(4) * 4294967296,
            size: blob.readUInt32LE(8)}
  }
}

function block_cache_t(memory_limit) {
  // block_t → uncompressed Buffer (may be empty to indicate pending)
  const lru = new lru_t({
    max: memory_limit,
    length: (data, block) => data.length,
  })

  // Do we have a block?
  this.contains = block => lru.has('' + block)

  // Add data for a block
  const set = async (block, compressed) => {
    if (Math.random() < flake_probability)
      throw Error('flaking for unit test purposes')
    const data = await lzma.decompress(compressed)
    const shape = block_shape(block[0], block[1])
    const expected = 64 * shape[0] * shape[1] * shape[2] * shape[3]
    if (data.length != expected)
      throw Error('data for section [' + block[0] + '], block [' + block[1] +
                  '] has length ' + data.length + ' != ' + expected)
    lru.set('' + block, uninterleave(data))
  }

  // Location information about a board
  const board_info = board => {
    if (board.middle)
      throw Error('board_info: middle board ' + board.name + ' not supported')

    // Account for global symmetries
    const [section, symmetry1] = standardize_section(board_section(board))
    const quads = global_transform_board(symmetry1, board)

    // Account for local symmetries
    const index = [0,0,0,0]
    let local_rotations = 0
    for (let i = 0; i < 4; i++) {
      const ir = tables.rotation_minimal_quadrants_inverse[quads[i]]
      index[i] = ir >> 2
      local_rotations |= (ir&3) << 2*i
    }

    // Collect location info
    const block = index.map(i => floor(i / block_size))
    const shape = block_shape(section, block)
    const I = index.map(i => i % block_size)
    const flat = I[3] + shape[3] * (I[2] + shape[2] * (I[1] + shape[1] * I[0]))
    return [section, block, flat, local_rotations]
  }

  // Look up the section and block containing a given board
  const board_block = board => {
    const [section, block, flat, local_rotations] = board_info(board)
    return [section, block]
  }

  // 1 if the player to move wins, 0 for tie, -1 if the player to move loses
  // Works only if !middle and !done.
  const value_helper = board => {
    const [section, block, flat, local_rotations] = board_info(board)
    const data = lru.get('' + [section, block])
    if (!data)
      throw Error('block_cache_t: section [' + section + '], block [' + block + '] missing')
    const turn = section_sum(section) & 1
    const bit = i => data[i >> 3] >> (i & 7) & 1
    const offset = flat * 512 + local_rotations
    return bit(offset + 256 * turn) - bit(offset + 256 * (1 - turn))
  }

  // 1 if the player to move wins, 0 for tie, -1 if the player to move loses
  const value = board => {
    if (board.done())
      return board.immediate_value()
    else if (!board.middle) {
      // If we're not at a half move, look up the result
      return value_helper(board)
    } else {
      let best = -1
      for (const move of board.moves()) {
        best = max(best, -value(move))
        if (best == 1)
          break
      }
      return best
    }
  }

  // Public methods
  this.set = set
  this.board_block = board_block
  this.value = value
}

// Primary exports
exports.descendent_sections = descendent_sections
exports.supertensor_index_t = supertensor_index_t
exports.block_cache_t = block_cache_t

// Unit test exports
exports.set_flaky = set_flaky
exports.board_section = board_section
exports.transform_section = transform_section
exports.standardize_section = standardize_section
exports.section_sum = section_sum
exports.section_shape = section_shape
exports.block_shape = block_shape
exports.global_transform_board = global_transform_board
exports.uninterleave = uninterleave
exports.section_sig = section_sig
