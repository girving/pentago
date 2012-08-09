Import('env Library Program Toplevel')

Toplevel('pentago','#.')

generated = ['gen/%s'%f for f in 'tables.h tables.cpp'.split()]
env.Command(generated,'precompute.py','./precompute.py --prefix ${TARGET.dir}')

env = env.Clone(use_mpi=1)
env.Append(CPPPATH=['.'],CXXFLAGS='-Wno-invalid-offsetof')
Library(env,'pentago',['other_core','lzma','snappy'],extra=['gen/tables.cpp'],skip=['endgame-mpi.cpp'])

env = env.Clone()
env.Append(LIBS=['pentago','other_core'])
if 0 and env['PLATFORM']=='posix':
  env.Append(LIBS=['tcmalloc']) # tcmalloc doesn't ever release memory to the OS, which is scary
Program(env,'endgame-mpi','mpi/endgame-mpi.cpp')
