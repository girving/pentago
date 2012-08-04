Import('env Library Program Toplevel')

Toplevel('pentago','#.')

generated = ['gen/%s'%f for f in 'tables.h tables.cpp'.split()]
env.Command(generated,'precompute.py','./precompute.py --prefix ${TARGET.dir}')

env = env.Clone(use_mpi=1)
env.Append(CPPPATH=['.'])
Library(env,'pentago',['other_core','lzma'],extra=['gen/tables.cpp'],skip=['endgame-mpi.cpp'])

env = env.Clone()
env.Append(LIBS=['pentago','other_core'])
if env['PLATFORM']=='posix':
  env.Append(LIBS=['tcmalloc'])
Program(env,'endgame-mpi','mpi/endgame-mpi.cpp')
