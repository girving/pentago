// Perfect pentago explorer

'use strict'
var d3 = require('d3')
var board_t = require('./board.js').board_t
var server = 'http://localhost:8000'
console.log('server',server)

// Pull in math stuff
var pi = Math.PI
var cos = Math.cos
var sin = Math.sin
var max = Math.max
var sqrt = Math.sqrt

// Drawing parameters
var shrink = 1/2 // Shrink for debugging
var svg_size = 640*shrink
var center = svg_size/2
var scale = 420/6*shrink
var bar_size = .1
var spot_radius = .4
var rotator_radius = 2.5
var rotator_thickness = .2
var rotator_arrow = .4
var select_radius = 4

function draw_base() {
  // Grab and resize svg
  var svg = d3.select('svg')
    .attr('width',svg_size)
    .attr('height',svg_size)
    .append('g')
    .attr('transform','translate('+center+','+center+') scale('+scale+','+-scale+') ')

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
  var quads = svg.selectAll('g').data(qdata).enter().append('g')
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
      svg.select('#quadrant'+qx+qy).selectAll('a#spot').data(grid).enter().append('a')
        .attr('id','spot').append('circle')
        .attr('cx',function (d) { return d.x%3-1 })
        .attr('cy',function (d) { return d.y%3-1 })
        .attr('r',spot_radius)
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
        rotators.push({'path':path,'select':select,'qx':qx,'qy':qy,'d':d})
      }
  function spin(d) {
    // Animate our quadrant left or right by pi/2
    var angle = d.d*pi/2
    svg.select('#quadrant'+d.qx+d.qy)
      .transition().duration(500)
      .attrTween('transform',function (d,i,a) { return function (t) {
        var a = angle*(t-1)
        var shift = 3*(sqrt(2)*max(cos(a+pi/4),cos(a-pi/4))-1)
        return 'translate('+(d.cx+shift*(d.qx-1/2))+','+(d.cy+shift*(d.qy-1/2))+') rotate('+180/pi*a+') ' }})
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

  return {'svg':svg,'grid':grid}
}

function draw_board(svg,board) {
  // Update circles
  var classes = {0:board.middle?'empty':board.turn?'emptywhite':'emptyblack',1:'black',2:'white'}
  svg.selectAll('a#spot')
    .attr('xlink:href',board.middle ? null : function (d) {
      return board.grid[6*d.x+d.y] ? null : '#'+board.place(d.x,d.y).name })
    .select('circle')
    .attr('class',function (d) { return classes[board.grid[6*d.x+d.y]] })

  // Update rotators
  svg.selectAll('a#rotate')
    .attr('xlink:href',board.middle ? function (d) { return '#'+board.rotate(d.qx,d.qy,d.d).name } : null)
  svg.selectAll('.norotate,.rotateblack,.rotatewhite')
    .attr('class',board.middle ? board.turn ? 'rotatewhite' : 'rotateblack' : 'norotate')
}

function update(svg) {
  if (window.location.hash) {
    try {
      var hash = window.location.hash.substr(1)
      var board = new board_t(hash)
    } catch (e) {
      var error = 'Invalid board '+hash+', error = '+e
      console.error(error)
      d3.select('#error').text(error)
      return
    }
  } else
    var board = new board_t([0,0,0,0],false)
  console.log('update',board.name)
  draw_board(svg,board)
}

function main() {
  var svg = draw_base().svg
  update(svg)
  window.onhashchange = function () { update(svg) }
}

// Toplevel
window.onload = main
