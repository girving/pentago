// Pentago explorer page logic, driving the static svg emitted by cc/site.cc
//
// All state lives in the url hash as a comma-separated move history.  This
// file only toggles classes and attributes on the generated dom and fills
// in board values from the baked-in seed, localStorage cache, backend, or
// local wasm midsolve.  Animation is pure css: rotating a quadrant sets a
// transitioned transform on the .quadrant group, while an inner group
// counter-rotates instantly so the already-updated board starts visually
// at the old orientation (see tex/transforms.tex for the y-up ancestor).

import { parse_board } from './board.js'
import { midsolve } from './mid_async.js'
import { get as cache_get, set as cache_set } from './cache.js'

const backend_url = 'https://us-central1-naml-148801.cloudfunctions.net/pentago/'

// Colors for each board value, taking care to be nice to colorblind folk
const value_colors = {'1': '#00ff00', '0': '#0000ff', '-1': '#ff0000'}

// Values for the empty board and all of its children, baked in at build time
// so that the first page load paints fully without any network.  Each bit is
// a value+1 (all such values are -1 or 0, except the winning first moves,
// which are the four center placements): mids in b.moves() order for the
// empty board b, and rots in m.moves() order within each one-stone middle
// board m, packed as hex, msb first.
const unpack = (hex, n, f) => {
  for (let i = 0; i < n; i++)
    f(i, parseInt(hex[i >> 2], 16) >> (3 - (i & 3)) & 1)
}
if (cache_get('0') === null) {
  cache_set('0', 1)
  const mids = parse_board('0').moves()
  unpack('7bfffffde', 36, (i, bit) => cache_set(mids[i].name, bit))  // values 0 or 1
  unpack('ff007fef00ff000000000000bf00c03000dff7000c0300fe000000000000ff00fbfd00ff', 288,
         (i, bit) => cache_set(mids[i >> 3].moves()[i & 7].name, bit - 1))  // values -1 or 0
}

// Static dom handles.  Spot and rotator links carry no identifying
// attributes; their meaning comes from document order, which matches the
// emission loops in cc/site.cc:
//   rotators: (q, d) for q in 0..3, d in (-1, 1) → q = i>>1, d = (i&1) ? 1 : -1
//   spots: quadrant loops then x,y within → s = 6x+y as below
const $ = id => document.getElementById(id)
const svg = $('board')
const status_el = $('status')
const backs = document.querySelectorAll('a.back')
const rots = [...svg.querySelectorAll('a.rot')]
const spots = [...svg.querySelectorAll('a:not(.rot)')]
const rot_q = i => i >> 1
const rot_d = i => (i & 1) ? 1 : -1
const spot_s = i => {
  const qx = i / 18 | 0, qy = (i / 9 | 0) & 1, w = i % 9
  return 6 * (3 * qx + (w / 3 | 0)) + 3 * qy + w % 3
}
const quads = [0, 1, 2, 3].map(q => $('q' + q))  // css-transitioned groups
const grids = [0, 1, 2, 3].map(q => $('g' + q))  // instantly counter-rotated content

// Swivel state: how far each quadrant has visually rotated, in quarter turns.
// Swivel accumulates forever; the counter-rotation keeps net rotation zero
// once each transition finishes.
const swivel = [0, 0, 0, 0]
const spinning = [false, false, false, false]
for (const [q, e] of quads.entries())
  e.addEventListener('transitionend', () => { spinning[q] = false; render() })
for (const [i, a] of rots.entries())
  a.addEventListener('click', () => {
    swivel[rot_q(i)] += rot_d(i)
    spinning[rot_q(i)] = true
    // The default link navigation then updates the hash, which redraws
  })

// Current board and an epoch counter to drop stale async value fills
let board = parse_board('0')
let epoch = 0

const history = () => (location.hash || '#0').slice(1).split(',')

const error = msg => {
  status_el.textContent = ''
  const d = document.createElement('div')
  d.id = 'error'
  d.textContent = msg
  status_el.append(d)
}

const loading = msg => {
  status_el.textContent = ''
  msg.split('').forEach((c, i) => {
    const d = document.createElement('div')
    d.className = 'load'
    d.style.animationDelay = 1.7 * i / msg.length + 's'
    d.textContent = c
    status_el.append(d)
  })
}

const status_lines = lines => {
  status_el.textContent = ''
  for (const line of lines) {
    status_el.append(line)
    status_el.append(document.createElement('br'))
  }
}

// Show or hide a value dot by setting css variables on its containing
// element, consumed by the .v/.rv rules in the generated stylesheet
const value_dot = (e, v) => {
  if (v === null || v === undefined) {
    e.style.removeProperty('--vd')
    e.style.removeProperty('--v')
  } else {
    e.style.setProperty('--vd', 'inline')
    e.style.setProperty('--v', value_colors[v])
  }
}

// Value of a board if we know it, else null.  Like the server, done boards
// know their own value.
const known = b => b.done ? b.value : cache_get(b.name)

// svg path for a five-in-a-row marker, in screen coordinates
const tweak = c => c - 2.5 + .05 * (c > 2 ? 1 : -1)
function five_path(f) {
  const x0 = tweak(f[0][0]), y0 = -tweak(f[0][1])
  const x1 = tweak(f[4][0]), y1 = -tweak(f[4][1])
  const dx = x1 - x0, dy = y1 - y0
  const s = .15 / 2 / Math.sqrt(dx * dx + dy * dy)
  const nx = s * dy, ny = -s * dx
  return `M${x0 + nx},${y0 + ny}L${x1 + nx},${y1 + ny}L${x1 - nx},${y1 - ny}L${x0 - nx},${y0 - ny}z`
}

const svgns = 'http://www.w3.org/2000/svg'
function draw_fives(b) {
  const g = $('fives')
  g.textContent = ''
  for (const f of b.fives) {
    if (f.some(([x, y]) => spinning[2 * (x / 3 | 0) + (y / 3 | 0)]))
      continue
    const color = b.grid[6 * f[0][0] + f[0][1]]
    const p = document.createElementNS(svgns, 'path')
    p.setAttribute('class', 'five')
    p.style.fill = color == 1 ? 'black' : 'white'
    p.setAttribute('d', five_path(f))
    g.append(p)
    if (color == 2)  // Restore the black outlines of white stones under a white stripe
      for (const [x, y] of f) {
        const c = document.createElementNS(svgns, 'circle')
        c.setAttribute('class', 'mask')
        c.setAttribute('cx', tweak(x))
        c.setAttribute('cy', -tweak(y))
        c.setAttribute('r', .39)
        g.append(c)
      }
  }
}

// Fill in value dots and the header label from whatever is known so far
function fill_values(b) {
  const v = known(b)
  value_dot($('turn'), v)
  $('hl').textContent =
    b.done ? {'1': 'wins!', '0': 'ties!', '-1': 'loses!'}[b.value]
    : v === null ? 'to play'
    : {'1': 'to win', '0': 'to tie', '-1': 'to lose'}[v]
  for (const [i, a] of spots.entries()) {
    const s = spot_s(i)
    if (b.middle || b.done || b.grid[s])
      value_dot(a, null)
    else
      value_dot(a, known(b.place(s / 6 | 0, s % 6)))
  }
  for (const [i, a] of rots.entries()) {
    if (b.middle && !b.done) {
      const q = rot_q(i)
      const w = known(b.rotate(q >> 1, q & 1, rot_d(i)))
      value_dot(a, w === null ? null : -w)
    } else
      value_dot(a, null)
  }
}

// Ensure values for b and its children are available, then fill them in
function lookup(b) {
  const e = epoch
  const has = x => known(x) !== null
  if (b.done || (has(b) && b.moves().every(has)))
    return
  const start = Date.now()
  const absorb = (op, values) => {
    for (const [raw, value] of Object.entries(values))
      cache_set(parse_board(raw).name, value)
    if (epoch == e) {
      fill_values(b)
      status_lines([op + ' ' + b.count + ' stone board', 'elapsed = ' + (Date.now() - start) / 1000 + ' s'])
    }
  }
  if (b.count <= 17) {  // Look up via server
    loading('Looking up ' + b.count + ' stone board...')
    fetch(backend_url + b.name).then(async res => {
      if (!res.ok) throw Error('Server request failed, https status = ' + res.status)
      absorb('Received', await res.json())
    }).catch(err => { if (epoch == e) error(err.message) })
  } else {  // Compute locally via WebAssembly
    loading('Computing ' + b.count + ' stone board locally...')
    midsolve(b).then(values => absorb('Computed', values))
      .catch(err => { if (epoch == e) error(err.message) })
  }
}

function render() {
  epoch++
  const hist = history()
  try {
    board = parse_board(hist[hist.length - 1])
  } catch (err) {
    error('Invalid board ' + location.hash + ', error = ' + err.message)
    return
  }
  const b = board
  const base = '#' + hist.join(',') + ','
  const mid = b.middle && !b.done

  // Turn-dependent css state and header stone
  svg.setAttribute('class', (b.turn ? 'wturn' : 'bturn') + (mid ? ' mid' : ''))
  $('turn').style.setProperty('--f', b.turn ? 'white' : 'black')

  // Back links
  const back = hist.length > 1 ? '#' + hist.slice(0, -1).join(',') : null
  for (const a of backs)
    back ? a.setAttribute('href', back) : a.removeAttribute('href')

  // Quadrant swivels.  A move with d=+1 turns the board contents 90°
  // counterclockwise on screen, so the animated group sweeps to -90*swivel
  // while the inner group counter-rotates to +90*swivel, making the new
  // board start visually at the old orientation and settle at net zero.
  for (const q of [0, 1, 2, 3]) {
    quads[q].style.transform = `rotate(${-90 * swivel[q]}deg)`
    grids[q].setAttribute('transform', `rotate(${90 * swivel[q]})`)
  }

  // Stones and placement links
  for (const [i, a] of spots.entries()) {
    const s = spot_s(i)
    const v = b.grid[s]
    const q = 2 * (s / 18 | 0) + (s % 6 / 3 | 0)
    const open = !v && !b.middle && !b.done
    a.setAttribute('class', v ? v == 1 ? 'b' : 'w' : open && !spinning[q] ? 'p' : '')
    if (open)
      a.setAttribute('href', base + b.place(s / 6 | 0, s % 6).name)
    else
      a.removeAttribute('href')
  }

  // Rotation links
  for (const [i, a] of rots.entries()) {
    const q = rot_q(i)
    if (mid)
      a.setAttribute('href', base + b.rotate(q >> 1, q & 1, rot_d(i)).name)
    else
      a.removeAttribute('href')
  }

  draw_fives(b)

  // Status and values
  if (b.done)
    status_lines(['Game complete',
                  b.value ? (b.value > 0) == b.turn ? 'White wins!' : 'Black wins!' : 'Tie!'])
  else
    status_el.textContent = ''
  fill_values(b)
  lookup(b)
}

// Favicon: a decorative board position, drawn with the same parameters as
// cc/svgs.cc favicon (which still generates the standalone favicon.svg used
// by details.html).  b/w are stones, v/t are win/tie value dots on tan.
{
  const cells = 'bwwtbtwttbvtvbwtvbtbvtwtvtwbbvtwbwwt'
  let s = ''
  cells.split('').forEach((c, i) => {
    const x = 10 * (i % 6) + 5, y = 10 * (i / 6 | 0) + 5
    const f = c == 'b' ? 'black' : c == 'w' ? 'white' : 'tan'
    s += `<circle cx="${x}" cy="${y}" r="4" fill="${f}" stroke="black" stroke-width=".3"/>`
    if (c == 'v' || c == 't')
      s += `<circle cx="${x}" cy="${y}" r="2" fill="${c == 'v' ? 'green' : 'blue'}"/>`
  })
  const link = document.createElement('link')
  link.rel = 'icon'
  link.href = 'data:image/svg+xml,' + encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60"><rect width="60" height="60" fill="tan"/>${s}</svg>`)
  document.head.append(link)
}

window.onhashchange = render
render()

// Test boards:
//   base: #0
//   rotation and fives: #238128874881424344m
//   white wins: #3694640587299947153m
//   black wins: #3694640600154188633m
//   tie: #3005942238600111847
//   midsolve: #274440791932540184
