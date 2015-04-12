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
      'conditions': [
        ['OS=="linux"', {
          'cflags_cc': ['-std=c++11'],
          'cflags_cc!': ['-fno-rtti','-fno-exceptions'],
          'libraries': ['/usr/local/lib/libpentago_core.so'],
        }],
        ['OS=="mac"', {
          'xcode_settings': {
            'OTHER_CPLUSPLUSFLAGS': ['-std=c++11'],
            'GCC_ENABLE_CPP_RTTI': 'YES',
            'GCC_ENABLE_CPP_EXCEPTIONS': 'YES',
            'MACOSX_DEPLOYMENT_TARGET':'10.9',
          },
          'libraries': ['/usr/local/lib/libpentago_core.dylib'],
        }]
      ]
    }
  ]
}
