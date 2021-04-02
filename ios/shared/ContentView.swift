// Content view

import SwiftUI

struct ContentView: View {
  /*
  let blah = (0..<0).map { i in MidSolver().solve(MidTest().board) }
  var body: some View {
    Text("Hello, world!\n\(blah.description)")
      .padding()
  }
  */
  @State var board: Board = try! Board("274440791932540184")
  let boardValue: Optional<Int> = .none

  var turn: Stone { board.turnStone }
  var turnLabel: String {
    switch boardValue {
    case .none: return "to play"
    case .some(1): return "to win"
    case .some(0): return "to tie"
    case .some(-1): return "to lose"
    default: return ""
    }
  }

  let squareSize: CGFloat = 40
  var barSize: CGFloat { 0.1 * squareSize }
  var spotRadius: CGFloat { 0.4 * squareSize }
  var valueRadius: CGFloat { 0.15 * squareSize }
  var rotatorRadius: CGFloat { 2.5 * squareSize }
  var rotatorThickness: CGFloat { 0.2 * squareSize }
  var rotatorArrow: CGFloat { 0.4 * squareSize }

  let barColor = Color.gray
  let boardColor = Color(red: 0.82, green: 0.70, blue: 0.55)
  let valueColors: [Int: Color] = [1: .green, 0: .blue, -1: .red]

  func color(_ s: Stone) -> Color {
    switch (s) {
    case .empty: return .clear
    case .white: return .white
    case .black: return .black
    }
  }

  // Quadrant data
  struct Square {
    let q: Int
    let k: Int
    var x: Int { k / 3 }
    var y: Int { k % 3 }
    subscript(_ board: Board) -> Stone { board.stone(q: q, k: k) }
    func place(_ board: Board) -> Board? { board.place(q: q, k: k) }
  }
  struct Quadrant {
    let q: Int
    var x: Int { q >> 1 }
    var y: Int { q & 1 }
    var grid: [Square] { (0..<9).map { Square(q: q, k: $0) } }
  }
  let quads: [Quadrant] = (0..<4).map { Quadrant(q: $0) }

  func spot(_ stone: Stone, _ value: Int?) -> some View {
    ZStack {
      Circle()
        .fill(color(stone))
        .frame(width: 2*spotRadius, height: 2*spotRadius)
      Circle()
        .stroke(Color.black)
        .frame(width: 2*spotRadius, height: 2*spotRadius)
      Circle()
        .fill(value.flatMap { valueColors[$0] } ?? Color.clear)
        .frame(width: 2*valueRadius, height: 2*valueRadius)
    }
  }

  func rotator(_ q: Quadrant, _ left: Bool, out: Bool) -> some Shape {
    let r = rotatorRadius
    let a = rotatorArrow
    let h = rotatorThickness / 2
    let qt = atan2(0.5 - Double(q.y), Double(q.x) - 0.5)
    let dt: Double = left ? 1 : -1
    let t0 = qt + 0.06*dt
    let t1 = qt + .pi/4*dt
    let t2 = t1 + Double(a/r)*dt
    let v0 = Angle(radians: t0 + 0.2*(t1-t0))
    let v1 = Angle(radians: t0 + 0.8*(t1-t0))
    func path(_ rect: CGRect) -> Path {
      let c = CGPoint(x: rect.midX, y: rect.midY)
      func point(_ r: CGFloat, _ t: Double) -> CGPoint {
        CGPoint(x: c.x + r*CGFloat(cos(t)), y: c.y + r*CGFloat(sin(t)))
      }
      var p = Path()
      if out {
        p.addArc(center: c, radius: r-h, startAngle: .radians(t0), endAngle: .radians(t1), clockwise: !left)
        p.addLine(to: point(r-a, t1))
        p.addLine(to: point(r, t2))
        p.addLine(to: point(r+a, t1))
        p.addArc(center: c, radius: r+h, startAngle: .radians(t1), endAngle: .radians(t0), clockwise: left)
        p.closeSubpath()
      } else {
        p.addArc(center: c, radius: r-h, startAngle: v0, endAngle: v1, clockwise: !left)
        p.addArc(center: c, radius: r+h, startAngle: v1, endAngle: v0, clockwise: left)
        p.closeSubpath()
      }
      return p
    }
    struct Rotator: Shape {
      var p: (CGRect) -> Path
      func path(in rect: CGRect) -> Path {
        p(rect)
      }
    }
    return Rotator(p: path)
  }

  var body: some View {
    VStack {
      // Header
      spot(turn, boardValue)
      Text(turnLabel)

      // Board
      ZStack {
        // Separators
        ForEach(0..<2) { n in
          Rectangle()
            .fill(barColor)
            .frame(width: barSize + 6.2*squareSize*CGFloat(n),
                   height: barSize + 6.2*squareSize*CGFloat(1-n))
        }

        // Quadrants
        ForEach(quads, id: \.q) { q in
          ZStack {
            if board.middle {
              ForEach(0..<2) { l in
                ZStack {
                  rotator(q, l>0, out: true).fill(boardColor)
                  rotator(q, l>0, out: true).stroke(Color.black)
                  rotator(q, l>0, out: false).fill(Color.purple)
                  rotator(q, l>0, out: false).stroke(Color.black)
                }.frame(width: 3*squareSize, height: 3*squareSize)
              }
            }
            Rectangle()
              .fill(boardColor)
              .frame(width: 3*squareSize, height: 3*squareSize)
            ForEach(q.grid, id: \.k) { g in
              spot(g[board], .none)
                .offset(x: squareSize*CGFloat(g.x - 1),
                        y: squareSize*CGFloat(1 - g.y))
                .onTapGesture {
                  print("try place: q \(g.q), k \(g.k)")
                  if let b = g.place(board) {
                    print("  success")
                    board = b
                  }
                }
            }
          }.offset(x: (barSize/2 + 1.5*squareSize) * (2*CGFloat(q.x) - 1),
                   y: (barSize/2 + 1.5*squareSize) * (1 - 2*CGFloat(q.y)))
        }
      }
    }.toolbar {
      ToolbarItem(placement: ToolbarItemPlacement.navigation) {
        Button(action: {
          print("Back!")
        }) {
          Label("Back", systemImage: "chevron.left")
        }
      }
    }
  }
}

struct ContentView_Previews: PreviewProvider {
  static var previews: some View {
    ContentView()
  }
}
