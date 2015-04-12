// Perfect pentago explorer

'use strict'
var d3 = require('d3')
var LRU = require('lru-cache')
var board_t = require('./board.js').board_t

// Pull in math stuff
var pi = Math.PI
var cos = Math.cos
var sin = Math.sin
var max = Math.max
var sqrt = Math.sqrt
var floor = Math.floor

// Backend, with a bit of caching to avoid flicker on the back button
var backend_url = 'https://backend.perfect-pentago.net:2048/'
var cache = LRU({max:2048})

// Colors for each board value, taking care to be nice to colorblind folk.
var value_colors = {'1':'#00ff00','0':'#0000ff','-1':'#ff0000','undefined':null}

// Which quadrant is currently spinning?
var spinning = null

// Drawing parameters
var bar_size = .1
var spot_radius = .4
var header_size = 2.5
var footer_size = 3.5
var margin_size = 1.7

function resize() {
  var root = d3.select('svg#board')
  var width = parseInt(root.style('width').match(/^(\d+)px$/)[1])
  var scale = width/(6+2*margin_size)
  var svg = root
    .attr('width',width)
    .attr('height',scale*(6+header_size+footer_size))
    .select('g')
    .attr('transform','translate('+scale*(3+margin_size)+','+scale*(3+header_size)+') scale('+scale+','+-scale+') ')
}

// Webkit has a strange bug which causes the turn label to disappear whenever
// we change the text.  Work around this by creating all the text in advance
// and making only the one we want visible.
var set_turnlabel = null // Call to set turnlabel text

function draw_base() {
  // Drawing parameters
  var value_radius = .15
  var rotator_radius = 2.5
  var rotator_thickness = .2
  var rotator_arrow = .4
  var select_radius = 4
  var font_size = .4

  // Grab and resize svg
  var svg = d3.select('svg#board').append('g')
  window.onresize = resize
  resize()

  // Draw header
  var header_y = 4.5
  svg.selectAll('circle').data([1,0]).enter().append('circle')
    .attr('class',function (d) { return d ? 'empty' : 'tvalue' })
    .attr('id',function (d) { return d ? 'turn' : null })
    .attr('cx',0)
    .attr('cy',header_y)
    .attr('r',function (d) { return d ? spot_radius : value_radius })

  // Draw all turnlabels, but make them initially invisible
  var labels = ['wins!','ties!','loses!','to win','to tie','to lose','to play']
  var turnlabels = svg.selectAll('text').data(labels).enter().append('text')
    .attr('class','turnlabel')
    .attr('transform','scale(1,-1)')
    .attr('x',0)
    .attr('y',-(header_y-spot_radius-font_size))
    .style('font-size',font_size)
    .style('visibility','hidden')
    .text(function (d) { return d })
  set_turnlabel = function (text) {
    turnlabels.style('visibility',function (d) { return d==text ? 'visible' : 'hidden' })
  }

  // Draw footer
  var footer_sep = 1.5
  var footer_cy = -5
  var footer_radius = .25
  var footer = svg.selectAll('.footer').data([1,0,-1]).enter().append('g')
    .attr('class','footer')
  footer.append('circle')
    .attr('class','fvalue')
    .attr('cx',function (d) { return -footer_sep*d })
    .attr('cy',footer_cy)
    .attr('r',footer_radius)
    .attr('fill',function (d) { return value_colors[d] })
  footer.append('text')
    .attr('class','valuelabel')
    .attr('transform','scale(1,-1)')
    .attr('x',function (d) { return -footer_sep*d })
    .attr('y',-(footer_cy-footer_radius-font_size))
    .text(function (d) { return {'1':'win','0':'tie','-1':'loss'}[d] })
    .style('font-size',font_size)

  // Draw separators
  svg.selectAll('rect').data([0,1]).enter().append('rect')
    .attr('class','separators')
    .attr('x',function (d) { return -(bar_size+6.2* d)/2 })
    .attr('y',function (d) { return -(bar_size+6.2*!d)/2 })
    .attr('width' ,function (d) { return bar_size+6.2* d })
    .attr('height',function (d) { return bar_size+6.2*!d })

  // Draw quadrants
  var qdata = []
  for (var qx=0;qx<2;qx++)
    for (var qy=0;qy<2;qy++)
      qdata.push({'qx':qx,'qy':qy,
                  'cx':(bar_size+3)*(qx-1/2),
                  'cy':(bar_size+3)*(qy-1/2)})
  var quads = svg.selectAll('g.quadrant').data(qdata).enter().append('g')
    .attr('class','quadrant')
    .attr('id',function (d) { return 'quadrant'+d.qx+d.qy })
    .attr('transform',function (d) { return 'translate('+d.cx+','+d.cy+') ' })
  quads.append('rect')
    .attr('class','board')
    .attr('x',-1.5)
    .attr('y',-1.5)
    .attr('width',3)
    .attr('height',3)

  // Initialize circles
  for (var qx=0;qx<2;qx++)
    for (var qy=0;qy<2;qy++) {
      var grid = []
      for (var x=0;x<3;x++)
        for (var y=0;y<3;y++)
          grid.push({'qx':qx,'qy':qy,'x':3*qx+x,'y':3*qy+y})
      qdata[2*qx+qy]['grid'] = grid
      var links = svg.select('#quadrant'+qx+qy).selectAll('a#spot')
        .data(grid).enter().append('a').attr('id','spot')
      var make_circles = function (id,radius) {
        return links.append('circle').attr('id',id)
          .attr('cx',function (d) { return d.x%3-1 })
          .attr('cy',function (d) { return d.y%3-1 })
          .attr('r',radius)
      }
      make_circles('spot',spot_radius)
      make_circles('value',value_radius)
        .attr('class','cvalue')
    }

  // Initialize rotators
  var rotators = []
  for (var qx=0;qx<2;qx++)
    for (var qy=0;qy<2;qy++)
      for (var d=-1;d<=1;d+=2) {
        var dx = qx?1:-1
        var dy = qy?1:-1
        var cx = 3*qx+1-2.5+bar_size/2*dx
        var cy = 3*qy+1-2.5+bar_size/2*dy
        var r = rotator_radius
        if ((d>0)^(qx==qy))
          var xa = 0, ya = dy, xb = dx, yb = 0
        else
          var xa = dx, ya = 0, xb = 0, yb = dy
        var point = function (r,t) {
          var c = cos(t)
          var s = sin(t)
          return [cx+r*(c*xa+s*xb),cy+r*(c*ya+s*yb)]
        }
        var a = rotator_arrow
        var h = rotator_thickness/2
        var t0 = .85, t1 = pi/2, t2 = t1+a/r
        var sa = select_radius
        var select =   'm'+point(0,0)+' L'+point(sa,t2)
                     +' A'+[sa,sa]+' 0 0 '+(d>0?0:1)+' '+point(sa,t0)
                     +' z'
        var path =   'm'+point(r-h,t0)
                   +' A'+[r-h,r-h]+' 0 0 '+(d>0?1:0)+' '+point(r-h,t1)
                   +' L'+point(r-a,t1)
                   +' L'+point(r,t2)
                   +' L'+point(r+a,t1)
                   +' L'+point(r+h,t1)
                   +' A'+[r+h,r+h]+' 0 0 '+(d>0?0:1)+' '+point(r+h,t0)
                   +' z'
        var v0 = t0+.2*(t1-t0),
            v1 = t0+.8*(t1-t0)
        var value =   'm'+point(r-h,v0)
                    +' A'+[r-h,r-h]+' 0 0 '+(d>0?1:0)+' '+point(r-h,v1)
                    +' L'+point(r+h,v1)
                    +' A'+[r+h,r+h]+' 0 0 '+(d>0?0:1)+' '+point(r+h,v0)
                    +' z'
        rotators.push({'path':path,'select':select,'value':value,'qx':qx,'qy':qy,'d':d})
      }
  function spin(d) {
    // Animate our quadrant left or right by pi/2
    var angle = d.d*pi/2
    spinning = [d.qx,d.qy]
    draw_fives(svg)
    svg.select('#quadrant'+d.qx+d.qy)
      .transition().duration(500)
      .attrTween('transform',function (d,i,a) { return function (t) {
        var a = angle*(t-1)
        var shift = 3*(sqrt(2)*max(cos(a+pi/4),cos(a-pi/4))-1)
        return 'translate('+(d.cx+shift*(d.qx-1/2))+','+(d.cy+shift*(d.qy-1/2))+') rotate('+180/pi*a+') ' }})
      .each('end', function () { if (spinning) { spinning = null; draw_fives(svg) } })
    d3.timer.flush() // Avoid flicker by starting transition immediately
  }
  var links = svg.selectAll('a#rotate').data(rotators).enter()
    .append('a')
    .attr('id','rotate')
    .on('click',spin)
  links.append('path')
    .attr('class','rotateselect')
    .attr('d',function (d) { return d.select })
  links.append('path')
    .attr('class','norotate')
    .attr('d',function (d) { return d.path })
  links.append('path')
    .attr('class','rvalue')
    .attr('d',function (d) { return d.value })

  return {'svg':svg,'grid':grid}
}

// Keep track of currently shown board for asynchronous callback purposes
var current_board = null

function draw_board(svg,board,history) {
  // Update turn
  svg.select('circle')
    .attr('class',board.turn ? 'white' : 'black')

  // Update back button
  d3.selectAll('a.back')
    .attr('href',history.length>1 ? '#'+history.slice(0,history.length-1).join(',') : null)

  // Update circles
  var classes = {0:board.middle?'empty':board.turn?'emptywhite':'emptyblack',1:'black',2:'white'}
  var base = '#'+history.join(',')+','
  var links = svg.selectAll('a#spot')
    .attr('xlink:href',board.middle ? null : function (d) {
      return board.grid[6*d.x+d.y] ? null : base+board.place(d.x,d.y).name})
  links.selectAll('circle#spot')
    .attr('class',function (d) { return classes[board.grid[6*d.x+d.y]] })

  // Update rotators
  svg.selectAll('a#rotate')
    .attr('xlink:href',board.middle ? function (d) { return base+board.rotate(d.qx,d.qy,d.d).name } : null)
  svg.selectAll('.rotateselect') // Hide rotate selectors from the mouse when they aren't active
    .style('pointer-events',board.middle ? null : 'none')
  svg.selectAll('.norotate,.rotateblack,.rotatewhite')
    .attr('class',board.middle ? board.turn ? 'rotatewhite' : 'rotateblack' : 'norotate')

  // Set status if we're done
  if (board.done()) {
    var v = board.immediate_value()
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
  var board = current_board
  var fives = board.fives()
  var active = []
  var masks = []
  for (var i=0;i<fives.length;i++) {
    var f = fives[i]
    var good = true
    if (spinning) {
      for (var j=0;j<5;j++)
        if (floor(f[j][0]/3)==spinning[0] && floor(f[j][1]/3==spinning[1])) {
          good = false
          break
        }
    }
    if (good) {
      active.push(f)
      if (board.grid[6*f[0][0]+f[0][1]]==2)
        for (var j=0;j<5;j++)
          masks.push(f[j])
    }
  }
  function tweak (c) {
    return c-2.5+bar_size/2*(c>2?1:-1)
  }
  var f = svg.selectAll('.five').data(active)
  f.enter().insert('path','.mask')
  f.exit().remove()
  f.attr('class', 'five')
   .style('fill', function (d) { return board.grid[6*d[0][0]+d[0][1]]==1 ? 'black' : 'white' })
   .attr('d', function (d) {
      var x0 = tweak(d[0][0]),
          y0 = tweak(d[0][1]),
          x1 = tweak(d[4][0]),
          y1 = tweak(d[4][1]),
          dx = x1-x0,
          dy = y1-y0,
          s = .15/2/sqrt(dx*dx+dy*dy),
          nx =  s*dy,
          ny = -s*dx
      function point(x,y) { return x+','+y }
      return 'm'+point(x0+nx,y0+ny)
           +' L'+point(x1+nx,y1+ny)
           +' L'+point(x1-nx,y1-ny)
           +' L'+point(x0-nx,y0-ny)
           +' z'
     })
  var m = svg.selectAll('.mask').data(masks)
  m.enter().append('circle')
  m.exit().remove()
  m.attr('class','mask')
   .attr('cx', function (d) { return tweak(d[0]) })
   .attr('cy', function (d) { return tweak(d[1]) })
   .attr('r', spot_radius-.01)
}

function draw_values(svg) {
  // Always draw values for the current board (set by draw_board above), even
  // if draw_values is called from an asynchronous callback.
  var board = current_board

  // Note if we're missing anything
  var missing = false
  var get = function (board) {
    var v = cache.get(board.name)
    if (v === undefined)
      missing = true
    return v
  }
  var has = function (board) {
    return !(get(board)===undefined)
  }

  // Draw values if we have them
  svg.selectAll('.cvalue')
    .style('opacity',board.middle || board.done() ? 0 : function (d) {
      return board.grid[6*d.x+d.y] || !has(board.place(d.x,d.y)) ? 0 : 1 })
    .style('fill',board.middle || board.done() ? null : function (d) {
      return board.grid[6*d.x+d.y] ? null : value_colors[get(board.place(d.x,d.y))] })
  svg.selectAll('.rvalue')
    .style('opacity',!board.middle || board.done() ? 0 : function (d) {
      return has(board.rotate(d.qx,d.qy,d.d)) ? 1 : 0 })
    .style('fill',!board.middle || board.done() ? null : function (d) {
      var v = get(board.rotate(d.qx,d.qy,d.d))
      return v===undefined ? null : value_colors[-v] })
  svg.selectAll('.tvalue')
    .style('opacity',board.done() || has(board) ? 1 : 0)
    .style('fill',value_colors[board.done() ? board.immediate_value() : get(board)])
  set_turnlabel(  board.done() ? {'1':'wins!','0':'ties!','-1':'loses!'}[board.immediate_value()]
                : has(board)   ? {'1':'to win','0':'to tie','-1':'to lose'}[get(board)]
                               : 'to play')

  // If we don't have them, look them up
  if (missing) {
    var xh = new XMLHttpRequest()
    var start = Date.now()
    xh.onreadystatechange = function () {
      if (xh.readyState==4) {
        if (xh.status==200) {
          var values = JSON.parse(xh.responseText)
          var elapsed = (Date.now()-start)/1000
          var s = 'Received '+board.count+' stone board<br>elapsed = '+elapsed+' s'
          if ('search-time' in values)
            s += ', tree search = '+values['search-time']+' s'
          set_status(s)
          for (var name in values)
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

function set_error(s) {
  console.log(s.replace('<br>',', '))
  d3.select('.status').html('<div id="'+mode+'">'+s+'</div>')
}

function set_loading(s) {
  console.log(s.replace('<br>',', '))
  var h = ''
  for (var i=0;i<s.length;i++) {
    var t = 1.7*i/s.length
    h += '<div class="load" style="animation-delay:'+t+'s;-webkit-animation-delay:'+t+'s">'+s[i]+'</div>'
  }
  d3.select('.status').html(h)
}

function update(svg) {
  var board = new board_t([0,0,0,0],false)
  var history = [board.name]
  if (window.location.hash) {
    try {
      var hash = window.location.hash.substr(1)
      history = hash.split(',')
      if (history.length)
        board = new board_t(history[history.length-1])
    } catch (e) {
      set_error('Invalid board '+hash+', error = '+e)
      return
    }
  }
  draw_board(svg,board,history)
}

function main() {
  var svg = draw_base().svg
  update(svg)
  window.onhashchange = function () { update(svg) }
}

// Toplevel
window.onload = main
