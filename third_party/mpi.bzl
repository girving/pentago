# Autodetect OpenMPI vs. MPICH


MPI_BUILD = """
package(default_visibility = ["//visibility:public"])

licenses(["restricted"])

cc_library(
    name = "mpi",
    srcs = ["{lib}"],
    hdrs = [{hdrs}],
    includes = ["."],
)
"""


def mpi_repository_impl(ctx):
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


mpi_repository = repository_rule(mpi_repository_impl)
