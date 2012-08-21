Import('env external library program toplevel')

toplevel('pentago','#.')
external('zlib',libs=['z'])
external('lzma',libs=['lzma'])
external('snappy',libs=['snappy'])

generated = ['gen/%s'%f for f in 'tables.h tables.cpp'.split()]
env.Command(generated,'precompute.py','./precompute.py --prefix ${TARGET.dir}')

env = env.Clone(use_mpi=1,use_zlib=1,use_lzma=1,use_snappy=1)
env.Append(CPPPATH=['.'],CXXFLAGS='-Wno-invalid-offsetof')
library(env,'pentago',['other_core'],extra=['gen/tables.cpp'],skip=['endgame-mpi.cpp'])

env = env.Clone()
env.Append(LIBS=['pentago','other_core'])
program(env,'endgame-mpi','mpi/endgame-mpi.cpp')
