# TensorFlow


TF_BUILD = """
package(default_visibility = ["//visibility:public"])

licenses(["restricted"])

cc_library(
    name = "tensorflow",
    srcs = [],
    hdrs = [{hdrs}],
    includes = ["."],
)
"""


def tf_repository_impl(ctx):
  # Collect headers
  base = '/usr/local/lib/python3.7/site-packages/tensorflow/include'
  find = ctx.execute(['find', base, '-name', '*.h', '-o', '-path', '*igen*', '-type', 'f'])
  hdrs = []
  for h in find.stdout.split('\n'):
    h = h.strip().replace(base + '/', '')
    if not h: continue
    ctx.symlink(base + '/' + h, h)
    hdrs.append('"' + h + '"')

  # Construct BUILD file
  ctx.file('BUILD', TF_BUILD.format(hdrs=', '.join(hdrs)))


tensorflow_repository = repository_rule(tf_repository_impl)
