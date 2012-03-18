Import('env Library')

generated = ['gen/%s'%f for f in 'tables.h tables.cpp'.split()]
env.Command(generated,'precompute','./precompute --prefix ${TARGET.dir}')

env = env.Clone()
env.Append(CPPPATH='.')
Library(env,'pentago',['other_core'],extra=['gen/tables.cpp'])
