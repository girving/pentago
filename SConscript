Import('env Library')

generated = ['gen/%s.h'%h for h in 'win rotate pack unpack move'.split()]
env.Command(generated,'precompute','./precompute --prefix ${TARGET.dir}')

env = env.Clone()
env.Append(CPPPATH='.')
Library(env,'pentago',['other_core'])
