"""GPT-2-based transformer."""

from dataclasses import dataclass
import inspect
import pentago
import numpy as np
import tensorflow as tf
from typing import Any, Tuple
layers = tf.keras.layers


@dataclass
class Params:
  dim: int = 256
  heads: int = 8
  layers: int = 4


@dataclass
class Spec:
  shape: Tuple[int,...]
  init: Any
  dtype: tf.DType = tf.float32


def shape_list(x):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class Simple(layers.Layer):
  def __init__(self, *, name, f, **specs):
    super(Simple, self).__init__(self, name=name)
    self._f = f
    self._specs = specs

  def build(self, input_shape):
    self._weights = {name: self.add_weight(name=name, dtype=s.dtype, shape=s.shape, initializer=s.init)
                     for name, s in self._specs.items()}
    super(Simple, self).build(input_shape)

  def call(self, x):
    return self._f(x, **self._weights)


def gelu(x):
  return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))
Gelu = Simple(name='gelu', f=gelu)


def Embedding2d(shape, dim, *, name='embed'):
  """tf.gather_nd embedding layer."""
  def f(X, we):
    i = tf.tile(tf.range(shape[0])[None], [X.shape[0], 1])
    return tf.gather_nd(we, tf.stack([i, X], axis=-1))
  return Simple(name=name, f=f, we=Spec(shape=(*shape,dim), init=tf.random_normal_initializer(stddev=0.01)))


def Norm(n_state, *, axis=-1, epsilon=1e-5, name):
  """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
  def f(x, g):
    s = tf.reduce_mean(tf.square(x), axis=axis, keepdims=True)
    return x * tf.rsqrt(s + epsilon) * g
  return Simple(name=name, f=f, g=Spec(shape=[n_state], init=tf.constant_initializer(1)))


def Linear(name, nin, nout, *, w_init_stdev=0.02):
  def f(x, w):
    *start, _ = shape_list(x)
    return tf.reshape(tf.matmul(tf.reshape(x, [-1, nin]), w), start+[nout])
  return Simple(name=name, f=f, w=Spec(shape=[nin, nout], init=tf.random_normal_initializer(stddev=w_init_stdev)))


def split_heads(x, *, H):
  # From [batch, sequence, features] to [batch, heads, sequence, features]
  assert x.shape.rank == 3
  *start, dim = shape_list(x)
  assert dim % H.heads == 0
  x = tf.reshape(x, start + [H.heads, dim//H.heads])
  return tf.transpose(x, [0, 2, 1, 3])


def merge_heads(x):
  # Reverse of split_heads
  x = tf.transpose(x, [0, 2, 1, 3])
  *start, a, b = shape_list(x)
  return tf.reshape(x, start + [a*b])


def multihead_attn(q, k, v):
  # q, k, v have shape [batch, heads, sequence, features]
  w = tf.matmul(q, k, transpose_b=True)
  w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))
  w = tf.nn.softmax(w)
  a = tf.matmul(w, v)
  return a


def Query(H):
  def query(c):
    q, k, v = (split_heads(x, H=H) for x in tf.split(c, 3, axis=2))
    return merge_heads(multihead_attn(q, k, v))
  return Simple(name='query', f=query)


def Attn(H):
  return tf.keras.Sequential(name='attn', layers=[
      Linear('c_attn', H.dim, H.dim*3),
      Query(H),
      Linear('c_proj', H.dim, H.dim)])


def FinalAttn(H):
  def f(c, q):
    batch, *rest = shape_list(c)
    q = split_heads(tf.tile(q[None], [batch, 1, 1]), H=H)
    k, v = (split_heads(x, H=H) for x in  tf.split(c, 2, axis=2))
    return merge_heads(multihead_attn(q, k, v))
  return tf.keras.Sequential(name='final_attn', layers=[
      Linear('c_attn', H.dim, H.dim*2),
      Simple(name='query', f=f, q=Spec(shape=(256,H.dim), init=tf.random_normal_initializer(stddev=0.01))),
      Linear('c_proj', H.dim, H.dim)])
   

def MLP(nx, n_state):
  return tf.keras.Sequential(name='mlp', layers=[
      Linear('c_fc', nx, n_state),
      Gelu,
      Linear('c_proj', n_state, nx)])


def Block(*, name, H):
  x = inputs = layers.Input(shape=(36, H.dim))
  a = Attn(H)(Norm(H.dim, name='ln_1')(x))
  x = x + a
  m = MLP(H.dim, H.dim*4)(Norm(H.dim, name='ln_2')(x))
  x = x + m
  return tf.keras.Model(inputs=inputs, outputs=x, name=name)


def Model(H):
  return tf.keras.Sequential(name='pentago', layers=[
      Embedding2d((36, 3), H.dim),
      *(Block(name='h%d' % n, H=H) for n in range(H.layers)),
      Norm(H.dim, name='ln_f'),
      FinalAttn(H),
      Linear('logits', H.dim, 3)])  # 3 cases are white-win, tie, black-win
