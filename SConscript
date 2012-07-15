Import('env Library')

generated = ['gen/%s'%f for f in 'tables.h tables.cpp'.split()]
env.Command(generated,'precompute.py','./precompute.py --prefix ${TARGET.dir}')

env = env.Clone(use_openmpi=1)
env.Append(CPPPATH=['.',Dir('#..').abspath])
Library(env,'pentago',['other_core','lzma'],extra=['gen/tables.cpp'],skip=['endgame-mpi.cpp'])
