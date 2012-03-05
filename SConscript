Import('env Library')

generated = ['gen/%s.h'%h for h in 'win rotate reflect pack unpack move rotated_win distance'.split()]
env.Command(generated,'precompute','./precompute --prefix ${TARGET.dir}')

env = env.Clone()
env.Append(CPPPATH='.')
Library(env,'pentago',['other_core'])
