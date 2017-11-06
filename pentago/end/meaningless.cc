#include "pentago/base/all_boards.h"
#include "pentago/base/symmetry.h"
#include "pentago/utility/log.h"

using namespace pentago;

int main() {
  string summary;
  for (const auto& board : all_boards(1, 8)) {
    const auto win = meaningless(board, 0);
    const auto win_or_tie = win | meaningless(board, 7);
    const char result = "ltw"[win+win_or_tie];
    if (0)
      slog("board %d - %d%d\n%s\n", board, win, win_or_tie, str_board(board));
    auto s = str_board(board);
    for (char& c : s)
      if (c == '0')
        c = result;
    if (summary.empty())
      summary = s;
    else {
      GEODE_ASSERT(summary.size() == s.size());
      for (const int i : range(int(s.size()))) {
        summary[i] = s[i]=='_' ? summary[i] : s[i];
      }
    }
  }

  slog("summary =\n%s", summary);
  return 0;
}
