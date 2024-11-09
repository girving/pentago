#include "pentago/utility/exceptions.h"
#include "pentago/utility/join.h"
#include "pentago/utility/log.h"
#include "pentago/utility/temporary.h"
#include "gtest/gtest.h"
#include <stdlib.h>

namespace pentago {
namespace {

using std::string;

const bool nop = false;

void run(const string& cmd) {
  slog(cmd);
  fflush(stdout);
  if (not nop) {
    setenv("HOME", "/not-a-real-directory", 0);  // OpenMPI insists on HOME being set, so fake it
    const int status = system(cmd.c_str());
    if (status)
      throw OSError(tfm::format("Command '%s' failed with status %d", cmd, status));
  }
}

void check(const string& dir, const string& options = "") {
  run(tfm::format("pentago/end/check %s %s", options, dir));
}

string mpirun() {
  if (nop)
    return "mpirun";
  const string cmds[] = {"mpirun", "aprun", "/usr/local/bin/mpirun"};
  for (const auto& cmd : cmds)
    if (!system(tfm::format("/usr/bin/which %s >/dev/null 2>/dev/null", cmd).c_str()))
      return cmd;
  throw OSError(tfm::format("No mpirun found, tried %s", join(", ", cmds)));
}

// Write out meaningless data from MPI
void write_test(const int slice) {
  tempdir_t tmp("write");
  const auto dir = tfm::format("%s/write-%d", tmp.path, slice);
  run(tfm::format("%s -n 2 pentago/mpi/endgame-mpi --threads 3 --dir %s --test write-%d",
                  mpirun(), dir, slice));
  check(dir, "--restart");
}
TEST(mpi, write_slice3) { write_test(3); }
TEST(mpi, write_slice4) { write_test(4); }

// Compute small count slices based on meaningless data
void meaningless_test(const int slice, const int key, const bool restart = false,
                      const bool extras = false) {
  tempdir_t tmp("meaningless");
  const auto wdir = tfm::format("%s/meangingless-s%d-r%d", tmp.path, slice, key);
  const auto base = tfm::format("%s -n 2 pentago/mpi/endgame-mpi --threads 3 --save 20 --memory 3G "
                                "--meaningless %d --randomize %d 00000000", mpirun(), slice, key);
  run(tfm::format("%s --dir %s", base, wdir));
  check(wdir);
  if (extras)
    check(wdir, tfm::format("--reader-test=%d --high-test=%d", slice-1, slice-2));
  if (restart) {
    const auto tdir = wdir + "-restart-test";
    run(tfm::format("%s --restart %s/slice-%d.pentago --dir %s --test restart",
                    base, wdir, slice-1, tdir));
    const auto rdir = wdir + "-restarted";
    run(tfm::format("%s --restart %s/slice-%d.pentago --dir %s", base, wdir, slice-1, rdir));
    check(rdir);
  }
}
TEST(mpi, meaningless_simple_slice4) { meaningless_test(4, 0); }
TEST(mpi, meaningless_simple_slice5) { meaningless_test(5, 0); }
TEST(mpi, meaningless_random_slice4) { meaningless_test(4, 17, true, true); }
TEST(mpi, meaningless_random_slice5) { meaningless_test(5, 17, true, true); }

}  // namespace
}  // namespace pentago
