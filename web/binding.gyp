{
  'make_global_settings': [
    ['CXX','/usr/bin/clang++']
  ],
  'targets': [
    {
      'target_name': 'pentago',
      'sources': ['pentago.cc'],
      'defines': [],
      'include_dirs': [
        '../bazel-pentago',
        '../bazel-genfiles'
      ],
      'libraries': [
        '../../bazel-bin/external/lzma/liblzma.a',
        '../../bazel-bin/external/zlib/libzlib.a',
        '../../bazel-bin/pentago/base/libbase.a',
        '../../bazel-bin/pentago/data/libasync.a',
        '../../bazel-bin/pentago/data/libdata.a',
        '../../bazel-bin/pentago/end/libend.a',
        '../../bazel-bin/pentago/high/libhigh.a',
        '../../bazel-bin/pentago/mid/libmid.a',
        '../../bazel-bin/pentago/utility/libutility.a'
      ],
      'conditions': [
        ['OS=="linux"', {
          'cflags_cc': ['-std=c++1z'],
          'cflags_cc!': ['-fno-rtti', '-fno-exceptions', '-Wdeprecated-declarations']
        }],
        ['OS=="mac"', {
          'xcode_settings': {
            'OTHER_CPLUSPLUSFLAGS': ['-std=c++1z', '-Wno-deprecated-declarations'],
            'GCC_ENABLE_CPP_RTTI': 'YES',
            'GCC_ENABLE_CPP_EXCEPTIONS': 'YES',
            'MACOSX_DEPLOYMENT_TARGET': '10.13',
          }
        }]
      ]
    }
  ]
}
