// Content view

import SwiftUI

struct ContentView: View {
  let blah = (0..<2).map { i in MidSolver().solve(MidTest().board19) }
  var body: some View {
    Text("Hello, world!\n\(blah.description)")
      .padding()
  }
}

struct ContentView_Previews: PreviewProvider {
  static var previews: some View {
    ContentView()
  }
}
