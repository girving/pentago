{
  'targets': [
    {
      'target_name': 'pentago',
      'sources': ['pentago.cc'],
      'defines': [],
      'include_dirs': [
        '..',
        '../bazel-pentago',
        '../bazel-pentago/external/boost',
        '../bazel-genfiles'
      ],
      'libraries': [
        '../../bazel-bin/pentago/mid/libmid.a',
        '../../bazel-bin/pentago/high/libhigh.a',
        '../../bazel-bin/pentago/end/libend.a',
        '../../bazel-bin/pentago/data/libasync.a',
        '../../bazel-bin/pentago/data/libdata.a',
        '../../bazel-bin/pentago/base/libbase.a',
        '../../bazel-bin/pentago/utility/libutility.a',
        '../../bazel-bin/external/lzma/liblzma.a',
        '../../bazel-bin/external/zlib/libzlib.a'
      ],
      'conditions': [
        ['OS=="linux"', {
          'cflags_cc': ['-std=c++1z', '-Wno-deprecated-declarations', '-Wno-implicit-fallthrough'],
          'cflags_cc!': ['-fno-rtti', '-fno-exceptions']
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
