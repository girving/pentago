# Autodetect OpenMPI vs. MPICH


MPI_BUILD = """
load("@rules_cc//cc:defs.bzl", "cc_library")
package(default_visibility = ["//visibility:public"])

licenses(["restricted"])

cc_library(
    name = "mpi",
    srcs = ["{lib}"],
    hdrs = [{hdrs}],
    includes = ["."],
)
"""


def _mpi_repository_impl(ctx):
  # Collect headers
  base = None
  for p in ['/usr/include', '/usr/local/include', '/opt/homebrew/include',
            '/opt/amazon/openmpi/include', '/opt/amazon/openmpi5/include']:
    p = ctx.path(p)
    if p.get_child('mpi.h').exists:
      base = p
  if base == None:
    fail('mpi.h not found')
  hdrs = []
  for h in ['mpi.h', 'mpio.h', 'mpi_portable_platform.h']:
    sh = base.get_child(h)
    if sh.exists:
      ctx.symlink(sh, h)
      hdrs.append('"' + h + '"')

  # Find the MPI shared library
  libbase = base.dirname.get_child('lib')
  lib = None
  for name in ['libmpi.dylib', 'libmpi.so']:
    if libbase.get_child(name).exists:
      lib = name
      break
  if lib == None:
    fail('libmpi not found under %s' % libbase)
  ctx.symlink(ctx.path(libbase.get_child(lib)), lib)

  # Construct BUILD file
  ctx.file('BUILD', MPI_BUILD.format(lib=lib, hdrs=', '.join(hdrs)))


_mpi_repository = repository_rule(_mpi_repository_impl)


def _mpi_ext_impl(ctx):
  _mpi_repository(name = "mpi")

mpi_ext = module_extension(implementation = _mpi_ext_impl)
