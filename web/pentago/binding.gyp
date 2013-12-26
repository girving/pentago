{
  'make_global_settings': [
    ['CXX','/usr/bin/clang++']
  ],
  'targets': [
    {
      'target_name': 'pentago',
      'sources': ['module.cpp'],
      'defines': ['GEODE_PYTHON','BOOST_EXCEPTION_DISABLE'],
      'include_dirs': ['/usr/include/python2.7'],
      'cflags_cc': ['-std=c++11'],
      'cflags_cc!': ['-fno-rtti','-fno-exceptions'],
      'libraries': ['/usr/local/lib/libpentago_core.so']
    }
  ]
}
