// Utilities

import Foundation

func large(_ n: Int) -> String {
  let formatter = NumberFormatter()
  formatter.numberStyle = .decimal
  return formatter.string(for: n) ?? ""
}
