// Perfect pentago explorer

'use strict'
const d3 = require('d3')
const LRU = require('lru-cache')
const board_t = require('./board.js').board_t

// Pull in math stuff
const pi = Math.PI
const cos = Math.cos
const sin = Math.sin
const max = Math.max
const sqrt = Math.sqrt
const floor = Math.floor

// Backend, with a bit of caching to avoid flicker on the back button
const backend_url = 'https://backend.perfect-pentago.net/'
const cache = LRU({max:2048})

// Colors for each board value, taking care to be nice to colorblind folk.
const value_colors = {'1':'#00ff00','0':'#0000ff','-1':'#ff0000','undefined':null}

// Which quadrant is currently spinning?
let spinning = null

// Drawing parameters
const bar_size = .1
const spot_radius = .4
const header_size = 2.5
const footer_size = 3.5
const margin_size = 1.7

function resize() {
  const root = d3.select('svg#board')
  const width = parseInt(root.style('width').match(/^(\d+)px$/)[1])
  const scale = width/(6+2*margin_size)
  root
    .attr('width',width)
    .attr('height',scale*(6+header_size+footer_size))
    .select('g')
    .attr('transform','translate('+scale*(3+margin_size)+','+scale*(3+header_size)+') scale('+scale+','+-scale+') ')
}

// Webkit has a strange bug which causes the turn label to disappear whenever
// we change the text.  Work around this by creating all the text in advance
// and making only the one we want visible.
let set_turnlabel = null  // Call to set turnlabel text

function draw_base() {
  // Drawing parameters
  const value_radius = .15
  const rotator_radius = 2.5
  const rotator_thickness = .2
  const rotator_arrow = .4
  const select_radius = 4
  const font_size = .4

  // Grab and resize svg
  const svg = d3.select('svg#board').append('g')
  window.onresize = resize
  resize()

  // Draw header
  const header_y = 4.5
  svg.selectAll('circle').data([1,0]).enter().append('circle')
    .attr('class', d => d ? 'empty' : 'tvalue')
    .attr('id', d => d ? 'turn' : null)
    .attr('cx', 0)
    .attr('cy', header_y)
    .attr('r', d => d ? spot_radius : value_radius)

  // Draw all turnlabels, but make them initially invisible
  const labels = ['wins!','ties!','loses!','to win','to tie','to lose','to play']
  const turnlabels = svg.selectAll('text').data(labels).enter().append('text')
    .attr('class','turnlabel')
    .attr('transform','scale(1,-1)')
    .attr('x',0)
    .attr('y',-(header_y-spot_radius-font_size))
    .style('font-size',font_size)
    .style('visibility','hidden')
    .text(d => d)
  set_turnlabel = text => {
    turnlabels.style('visibility',function (d) { return d==text ? 'visible' : 'hidden' })
  }

  // Draw footer
  const footer_sep = 1.5
  const footer_cy = -5
  const footer_radius = .25
  const footer = svg.selectAll('.footer').data([1,0,-1]).enter().append('g')
    .attr('class','footer')
  footer.append('circle')
    .attr('class', 'fvalue')
    .attr('cx', d => -footer_sep*d)
    .attr('cy', footer_cy)
    .attr('r', footer_radius)
    .attr('fill', d => value_colors[d])
  footer.append('text')
    .attr('class', 'valuelabel')
    .attr('transform', 'scale(1,-1)')
    .attr('x', d => -footer_sep*d)
    .attr('y', -(footer_cy-footer_radius-font_size))
    .text(d => ({'1':'win','0':'tie','-1':'loss'}[d]))
    .style('font-size', font_size)

  // Draw separators
  svg.selectAll('rect').data([0,1]).enter().append('rect')
    .attr('class','separators')
    .attr('x', d => -(bar_size+6.2* d)/2)
    .attr('y', d => -(bar_size+6.2*!d)/2)
    .attr('width',  d => bar_size+6.2* d)
    .attr('height', d => bar_size+6.2*!d)

  // Draw quadrants
  const qdata = []
  for (let qx=0;qx<2;qx++)
    for (let qy=0;qy<2;qy++)
      qdata.push({'qx':qx,'qy':qy,
                  'cx':(bar_size+3)*(qx-1/2),
                  'cy':(bar_size+3)*(qy-1/2)})
  const quads = svg.selectAll('g.quadrant').data(qdata).enter().append('g')
    .attr('class','quadrant')
    .attr('id', d => 'quadrant'+d.qx+d.qy)
    .attr('transform', d => 'translate('+d.cx+','+d.cy+') ')
  quads.append('rect')
    .attr('class','board')
    .attr('x',-1.5)
    .attr('y',-1.5)
    .attr('width',3)
    .attr('height',3)

  // Initialize circles
  for (let qx=0;qx<2;qx++)
    for (let qy=0;qy<2;qy++) {
      const grid = []
      for (let x=0;x<3;x++)
        for (let y=0;y<3;y++)
          grid.push({'qx':qx,'qy':qy,'x':3*qx+x,'y':3*qy+y})
      qdata[2*qx+qy]['grid'] = grid
      const links = svg.select('#quadrant'+qx+qy).selectAll('a#spot')
        .data(grid).enter().append('a').attr('id','spot')
      const make_circles = (id, radius) =>
        links.append('circle').attr('id', id)
          .attr('cx', d => d.x%3-1)
          .attr('cy', d => d.y%3-1)
          .attr('r', radius)
      make_circles('spot',spot_radius)
      make_circles('value',value_radius)
        .attr('class','cvalue')
    }

  // Initialize rotators
  const rotators = []
  for (let qx=0;qx<2;qx++)
    for (let qy=0;qy<2;qy++)
      for (let d=-1;d<=1;d+=2) {
        const dx = qx?1:-1
        const dy = qy?1:-1
        const cx = 3*qx+1-2.5+bar_size/2*dx
        const cy = 3*qy+1-2.5+bar_size/2*dy
        const r = rotator_radius
        let xa, ya, xb, yb
        if ((d>0)^(qx==qy))
          xa = 0, ya = dy, xb = dx, yb = 0
        else
          xa = dx, ya = 0, xb = 0, yb = dy
        const point = (r, t) => {
          const c = cos(t)
          const s = sin(t)
          return [cx+r*(c*xa+s*xb),cy+r*(c*ya+s*yb)]
        }
        const a = rotator_arrow
        const h = rotator_thickness/2
        const t0 = .85, t1 = pi/2, t2 = t1+a/r
        const sa = select_radius
        const select =   'm'+point(0,0)+' L'+point(sa,t2)
                       +' A'+[sa,sa]+' 0 0 '+(d>0?0:1)+' '+point(sa,t0)
                       +' z'
        const path =   'm'+point(r-h,t0)
                     +' A'+[r-h,r-h]+' 0 0 '+(d>0?1:0)+' '+point(r-h,t1)
                     +' L'+point(r-a,t1)
                     +' L'+point(r,t2)
                     +' L'+point(r+a,t1)
                     +' L'+point(r+h,t1)
                     +' A'+[r+h,r+h]+' 0 0 '+(d>0?0:1)+' '+point(r+h,t0)
                     +' z'
        const v0 = t0+.2*(t1-t0),
              v1 = t0+.8*(t1-t0)
        const value =   'm'+point(r-h,v0)
                      +' A'+[r-h,r-h]+' 0 0 '+(d>0?1:0)+' '+point(r-h,v1)
                      +' L'+point(r+h,v1)
                      +' A'+[r+h,r+h]+' 0 0 '+(d>0?0:1)+' '+point(r+h,v0)
                      +' z'
        rotators.push({'path':path,'select':select,'value':value,'qx':qx,'qy':qy,'d':d})
      }
  function spin(d) {
    // Animate our quadrant left or right by pi/2
    const angle = d.d*pi/2
    spinning = [d.qx,d.qy]
    draw_fives(svg)
    svg.select('#quadrant'+d.qx+d.qy)
      .transition().duration(500)
      .attrTween('transform', (d, i, a) => t => {
        const a = angle*(t-1)
        const shift = 3*(sqrt(2)*max(cos(a+pi/4),cos(a-pi/4))-1)
        return 'translate('+(d.cx+shift*(d.qx-1/2))+','+(d.cy+shift*(d.qy-1/2))+') rotate('+180/pi*a+') ' })
      .each('end', () => { if (spinning) { spinning = null; draw_fives(svg) }})
    d3.timer.flush() // Avoid flicker by starting transition immediately
  }
  const links = svg.selectAll('a#rotate').data(rotators).enter()
    .append('a')
    .attr('id','rotate')
    .on('click',spin)
  links.append('path')
    .attr('class', 'rotateselect')
    .attr('d', d => d.select)
  links.append('path')
    .attr('class', 'norotate')
    .attr('d', d => d.path)
  links.append('path')
    .attr('class', 'rvalue')
    .attr('d', d => d.value)

  return svg
}

// Keep track of currently shown board for asynchronous callback purposes
let current_board = null

function draw_board(svg, board, history) {
  // Update turn
  svg.select('circle')
    .attr('class', board.turn ? 'white' : 'black')

  // Update back button
  d3.selectAll('a.back')
    .attr('href', history.length>1 ? '#'+history.slice(0,history.length-1).join(',') : null)

  // Update circles
  const classes = {0:board.middle?'empty':board.turn?'emptywhite':'emptyblack',1:'black',2:'white'}
  const base = '#'+history.join(',')+','
  const links = svg.selectAll('a#spot')
    .attr('xlink:href', board.middle ? null : d =>
      board.grid[6*d.x+d.y] ? null : base+board.place(d.x,d.y).name)
  links.selectAll('circle#spot')
    .attr('class', d => classes[board.grid[6*d.x+d.y]])

  // Update rotators
  svg.selectAll('a#rotate')
    .attr('xlink:href', board.middle ? d => base+board.rotate(d.qx,d.qy,d.d).name : null)
  svg.selectAll('.rotateselect') // Hide rotate selectors from the mouse when they aren't active
    .style('pointer-events', board.middle ? null : 'none')
  svg.selectAll('.norotate,.rotateblack,.rotatewhite')
    .attr('class', board.middle ? board.turn ? 'rotatewhite' : 'rotateblack' : 'norotate')

  // Set status if we're done
  if (board.done()) {
    const v = board.immediate_value()
    set_status('Game complete<br>'+(v?(v>0)==board.turn?'White wins!':'Black wins!':'Tie!'))
  }

  // Draw win/loss/tie values
  current_board = board
  draw_values(svg)
  draw_fives(svg)
}

function draw_fives(svg) {
  // Always draw fives for the current board (set by draw_board above), even
  // if draw_values is called from an asynchronous callback.
  const board = current_board
  const fives = board.fives()
  const active = []
  const masks = []
  for (let i=0;i<fives.length;i++) {
    const f = fives[i]
    let good = true
    if (spinning) {
      for (let j=0;j<5;j++)
        if (floor(f[j][0]/3)==spinning[0] && floor(f[j][1]/3==spinning[1])) {
          good = false
          break
        }
    }
    if (good) {
      active.push(f)
      if (board.grid[6*f[0][0]+f[0][1]]==2)
        for (let j=0;j<5;j++)
          masks.push(f[j])
    }
  }
  function tweak (c) {
    return c-2.5+bar_size/2*(c>2?1:-1)
  }
  const f = svg.selectAll('.five').data(active)
  f.enter().insert('path','.mask')
  f.exit().remove()
  f.attr('class', 'five')
   .style('fill', d => board.grid[6*d[0][0]+d[0][1]]==1 ? 'black' : 'white')
   .attr('d', d => {
      const x0 = tweak(d[0][0]),
            y0 = tweak(d[0][1]),
            x1 = tweak(d[4][0]),
            y1 = tweak(d[4][1]),
            dx = x1-x0,
            dy = y1-y0,
            s = .15/2/sqrt(dx*dx+dy*dy),
            nx =  s*dy,
            ny = -s*dx
      const point = (x,y) => x+','+y
      return 'm'+point(x0+nx,y0+ny)
           +' L'+point(x1+nx,y1+ny)
           +' L'+point(x1-nx,y1-ny)
           +' L'+point(x0-nx,y0-ny)
           +' z'
     })
  const m = svg.selectAll('.mask').data(masks)
  m.enter().append('circle')
  m.exit().remove()
  m.attr('class','mask')
   .attr('cx', d => tweak(d[0]))
   .attr('cy', d => tweak(d[1]))
   .attr('r', spot_radius-.01)
}

function draw_values(svg) {
  // Always draw values for the current board (set by draw_board above), even
  // if draw_values is called from an asynchronous callback.
  const board = current_board

  // Note if we're missing anything
  let missing = false
  const get = board => {
    const v = cache.get(board.name)
    if (v === undefined)
      missing = true
    return v
  }
  const has = board => !(get(board)===undefined)

  // Draw values if we have them
  svg.selectAll('.cvalue')
    .style('opacity', board.middle || board.done() ? 0 : d =>
      board.grid[6*d.x+d.y] || !has(board.place(d.x,d.y)) ? 0 : 1)
    .style('fill', board.middle || board.done() ? null : d =>
      board.grid[6*d.x+d.y] ? null : value_colors[get(board.place(d.x,d.y))])
  svg.selectAll('.rvalue')
    .style('opacity', !board.middle || board.done() ? 0 : d =>
      has(board.rotate(d.qx,d.qy,d.d)) ? 1 : 0)
    .style('fill', !board.middle || board.done() ? null : d => {
      const v = get(board.rotate(d.qx,d.qy,d.d))
      return v===undefined ? null : value_colors[-v] })
  svg.selectAll('.tvalue')
    .style('opacity',board.done() || has(board) ? 1 : 0)
    .style('fill',value_colors[board.done() ? board.immediate_value() : get(board)])
  set_turnlabel(  board.done() ? {'1':'wins!','0':'ties!','-1':'loses!'}[board.immediate_value()]
                : has(board)   ? {'1':'to win','0':'to tie','-1':'to lose'}[get(board)]
                               : 'to play')

  // If we don't have them, look them up
  if (missing) {
    const xh = new XMLHttpRequest()
    const start = Date.now()
    xh.onreadystatechange = () => {
      if (xh.readyState==4) {
        if (xh.status==200) {
          const values = JSON.parse(xh.responseText)
          const elapsed = (Date.now()-start)/1000
          let s = 'Received '+board.count+' stone board<br>elapsed = '+elapsed+' s'
          if ('search-time' in values)
            s += ', tree search = '+values['search-time']+' s'
          set_status(s)
          for (const name in values)
            cache.set(name,values[name])
          draw_values(svg)
        } else
          set_error('Server request failed, https status = '+xh.status)
      }
    }
    set_loading('Looking up '+board.count+' stone board...')
    xh.open('GET',backend_url+board.name,true)
    xh.send()
  }
}

function set_status(s) {
  console.log(s.replace('<br>',', '))
  d3.select('.status').html(s)
}

function set_error(s, e) {
  console.log(s.replace('<br>',', '), e)
  d3.select('.status').html('<div id="error">'+s+'</div>')
}

function set_loading(s) {
  console.log(s.replace('<br>',', '))
  let h = ''
  for (let i=0;i<s.length;i++) {
    const t = 1.7*i/s.length
    h += '<div class="load" style="animation-delay:'+t+'s;-webkit-animation-delay:'+t+'s">'+s[i]+'</div>'
  }
  d3.select('.status').html(h)
}

function update(svg) {
  let board = new board_t([0,0,0,0], false)
  let history = [board.name]
  if (window.location.hash) {
    const hash = window.location.hash.substr(1)
    try {
      history = hash.split(',')
      if (history.length)
        board = new board_t(history[history.length-1])
    } catch (e) {
      set_error('Invalid board '+hash+', error = '+e.message, e)
      return
    }
  }
  draw_board(svg,board,history)
}

function main() {
  const svg = draw_base()
  update(svg)
  window.onhashchange = () => update(svg)
}

// Toplevel
window.onload = main
