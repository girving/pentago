Import('env child external library toplevel')

toplevel('pentago','#.')
external('zlib',libs=['z'])
external('lzma',libs=['lzma'])
external('snappy',libs=['snappy'])

child(env,'mpi')

generated = ['gen/%s'%f for f in 'tables.h tables.cpp'.split()]
env.Command(generated,'base/precompute.py','base/precompute.py --prefix ${TARGET.dir}')

env = env.Clone(use_zlib=1,use_lzma=1,use_snappy=1)
env.Append(CPPPATH=['.'],CXXFLAGS='-Wno-invalid-offsetof')
library(env,'pentago_core',['other_core'],extra=['gen/tables.cpp'],skip=['mpi'])
