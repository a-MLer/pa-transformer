# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CNN builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import contextlib

import numpy as np

import tensorflow.compat.v1 as tf

# pylint: disable=g-direct-tensorflow-import
import mlperf
from tensorflow.python.layers import convolutional as conv_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import normalization as normalization_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.training import moving_averages


_data_format_to_channel_axis = {'NCHW': 1, 'NHWC': 3}

def shape_list(x):
  static = x.get_shape().as_list()
  shape = tf.shape(x)
  ret = []
  for i, static_dim in enumerate(static):
    dim = static_dim or shape[i]
    ret.append(dim)
  return ret

def split_heads_2d(inputs, Nh):
  B, H, W, d = shape_list(inputs)
  ret_shape = [B, H, W, Nh, d // Nh]
  split = tf.reshape(inputs, ret_shape)
  return tf.transpose(split, [0, 3, 1, 2, 4])

def combine_heads_2d(inputs):
  transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])
  Nh, channels = shape_list(transposed)[-2:]
  ret_shape = shape_list(transposed)[:-2] + [Nh * channels]
  return tf.reshape(transposed, ret_shape)

def rel_to_abs(x):
  B, Nh, L, _ = shape_list(x)
  col_pad = tf.zeros((B, Nh, L, 1))
  x = tf.concat([x, col_pad], axis=3)
  flat_x = tf.reshape(x, [B, Nh, L*2*L])
  flat_pad = tf.zeros((B, Nh, L-1))
  flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)
  final_x = tf.reshape(flat_x_padded, [B, Nh, L+1, 2*L-1])
  final_x = final_x[:, :, :L, L-1:]
  return final_x

def relative_logits_1d(q, rel_k, H, W, Nh, transpose_mask):
  rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
  rel_logits = tf.reshape(rel_logits, [-1, Nh*H, W, 2*W-1])
  rel_logits = rel_to_abs(rel_logits)
  rel_logits = tf.reshape(rel_logits, [-1, Nh, H, W, W])
  rel_logits = tf.expand_dims(rel_logits, axis=3)
  rel_logits = tf.tile(rel_logits, [1, 1, 1, H, 1, 1])
  rel_logits = tf.transpose(rel_logits, transpose_mask)
  rel_logits = tf.reshape(rel_logits, [-1, Nh, H*W, H*W])
  return rel_logits

def relative_logits(q, H, W, Nh, dkh):
  rel_embeddings_w = tf.get_variable('r_width', shape=(2 * W - 1, dkh),
          initializer=tf.random_normal_initializer(dkh ** -0.5))
  rel_logits_w = relative_logits_1d(q, rel_embeddings_w, H, W, Nh, [0, 1, 2, 4, 3, 5])
  rel_embeddings_h = tf.get_variable('r_height', shape=(2 * H - 1, dkh),
          initializer=tf.random_normal_initializer(dkh ** -0.5))
  rel_logits_h = relative_logits_1d(tf.transpose(q, [0, 1, 3, 2, 4]),
          rel_embeddings_h, W, H, Nh, [0, 1, 4, 2, 5, 3])
  return rel_logits_h, rel_logits_w

class ConvNetBuilder(object):
  """Builder of cnn net."""

  def __init__(self,
               input_op,
               input_nchan,
               phase_train,
               use_tf_layers,
               data_format='NCHW',
               dtype=tf.float32,
               variable_dtype=tf.float32):
    self.top_layer = input_op
    self.top_size = input_nchan
    self.phase_train = phase_train
    self.use_tf_layers = use_tf_layers
    self.data_format = data_format
    self.dtype = dtype
    self.variable_dtype = variable_dtype
    self.counts = defaultdict(lambda: 0)
    self.use_batch_norm = False
    self.batch_norm_config = {}  # 'decay': 0.997, 'scale': True}
    self.channel_pos = ('channels_last'
                        if data_format == 'NHWC' else 'channels_first')
    self.aux_top_layer = None
    self.aux_top_size = 0
    self.prev_att_out = None

  def get_custom_getter(self):
    """Returns a custom getter that this class's methods must be called under.

    All methods of this class must be called under a variable scope that was
    passed this custom getter. Example:

    ```python
    network = ConvNetBuilder(...)
    with tf.variable_scope('cg', custom_getter=network.get_custom_getter()):
      network.conv(...)
      # Call more methods of network here
    ```

    Currently, this custom getter only does anything if self.use_tf_layers is
    True. In that case, it causes variables to be stored as dtype
    self.variable_type, then casted to the requested dtype, instead of directly
    storing the variable as the requested dtype.
    """
    def inner_custom_getter(getter, *args, **kwargs):
      """Custom getter that forces variables to have type self.variable_type."""
      if not self.use_tf_layers:
        return getter(*args, **kwargs)
      requested_dtype = kwargs['dtype']
      if not (requested_dtype == tf.float32 and
              self.variable_dtype == tf.float16):
        # Only change the variable dtype if doing so does not decrease variable
        # precision.
        kwargs['dtype'] = self.variable_dtype
      var = getter(*args, **kwargs)
      # This if statement is needed to guard the cast, because batch norm
      # assigns directly to the return value of this custom getter. The cast
      # makes the return value not a variable so it cannot be assigned. Batch
      # norm variables are always in fp32 so this if statement is never
      # triggered for them.
      if var.dtype.base_dtype != requested_dtype:
        var = tf.cast(var, requested_dtype)
      return var
    return inner_custom_getter

  @contextlib.contextmanager
  def switch_to_aux_top_layer(self):
    """Context that construct cnn in the auxiliary arm."""
    if self.aux_top_layer is None:
      raise RuntimeError('Empty auxiliary top layer in the network.')
    saved_top_layer = self.top_layer
    saved_top_size = self.top_size
    self.top_layer = self.aux_top_layer
    self.top_size = self.aux_top_size
    yield
    self.aux_top_layer = self.top_layer
    self.aux_top_size = self.top_size
    self.top_layer = saved_top_layer
    self.top_size = saved_top_size

  def get_variable(self, name, shape, dtype, cast_dtype, *args, **kwargs):
    # TODO(reedwm): Currently variables and gradients are transferred to other
    # devices and machines as type `dtype`, not `cast_dtype`. In particular,
    # this means in fp16 mode, variables are transferred as fp32 values, not
    # fp16 values, which uses extra bandwidth.
    var = tf.get_variable(name, shape, dtype, *args, **kwargs)
    return tf.cast(var, cast_dtype)

  def _conv2d_impl(self, input_layer, num_channels_in, filters, kernel_size,
                   strides, padding, kernel_initializer):
    if self.use_tf_layers:
      return conv_layers.conv2d(input_layer, filters, kernel_size, strides,
                                padding, self.channel_pos,
                                kernel_initializer=kernel_initializer,
                                use_bias=False)
    else:
      weights_shape = [kernel_size[0], kernel_size[1], num_channels_in, filters]
      # We use the name 'conv2d/kernel' so the variable has the same name as its
      # tf.layers equivalent. This way, if a checkpoint is written when
      # self.use_tf_layers == True, it can be loaded when
      # self.use_tf_layers == False, and vice versa.
      weights = self.get_variable('conv2d/kernel', weights_shape,
                                  self.variable_dtype, self.dtype,
                                  initializer=kernel_initializer)
      if self.data_format == 'NHWC':
        strides = [1] + strides + [1]
      else:
        strides = [1, 1] + strides
      return tf.nn.conv2d(input_layer, weights, strides, padding,
                          data_format=self.data_format)

  def asconv(self,
             num_out_channels,
             k_height,
             k_width,
             d_height=1,
             d_width=1,
             mode='SAME',
             alpha=0.5,
             k=0.25,
             v=0.125,
             Nh=8,
             do_downsample=False,
             relative=True,
             input_layer=None,
             num_channels_in=None,
             use_batch_norm=None,
             stddev=None,
             activation='relu',
             bias=0.0,
             kernel_initializer=None):
    """Construct a conv2d layer on top of cnn."""
    if input_layer is None:
      input_layer = self.top_layer
    if num_channels_in is None:
      num_channels_in = self.top_size
    if stddev is not None and kernel_initializer is None:
      kernel_initializer = tf.truncated_normal_initializer(stddev=stddev)
    if kernel_initializer is None:
      kernel_initializer = tf.variance_scaling_initializer()
    name = 'asconv' + str(self.counts['asconv'])
    self.counts['asconv'] += 1

    dk = int(num_out_channels * k)
    dv = int(num_out_channels * v)

    conv_out = self._conv2d_impl(input_layer, num_channels_in,
                                 num_out_channels - dv,
                                 kernel_size=[k_height, k_width],
                                 strides=[d_height, d_width], padding='SAME',
                                 kernel_initializer=kernel_initializer)

    with tf.variable_scope(name):
      if do_downsample:
        if self.data_format == "NCHW":
          _, _, orgH, orgW = shape_list(input_layer)
        else:
          _, orgH, orgW, _ = shape_list(input_layer)
        with tf.variable_scope("asconv_downsampling"):
          input_layer = tf.nn.avg_pool(input_layer, 3, 2, padding="SAME", data_format=self.data_format)

      # use NHWC for MHA
      if self.data_format == "NCHW":
        input_layer = tf.transpose(input_layer, [0, 2, 3, 1])

      _, H, W, _ = shape_list(input_layer)

      dkh = dk // Nh
      dvh = dv // Nh

      flatten_hw = lambda x, d: tf.reshape(x, [-1, Nh, H*W, d])

      with tf.variable_scope("att-proj"):
        kqv = tf.layers.conv2d(input_layer,
                2 * dk + dv, 1, kernel_initializer=kernel_initializer)

      k, q, v = tf.split(kqv, [dk, dk, dv], axis=3)
      q *= dkh ** -0.5

      q = split_heads_2d(q, Nh)
      k = split_heads_2d(k, Nh)
      v = split_heads_2d(v, Nh)

      logits = tf.matmul(flatten_hw(q, dkh), flatten_hw(k, dkh), transpose_b=True)

      if relative:
        rel_logits_h, rel_logits_w = relative_logits(q, H, W, Nh, dkh)
        logits += rel_logits_h
        logits += rel_logits_w

      att_input = self.prev_att_out
      # N C H W
      if att_input is not None \
         and logits.shape[1] == att_input.shape[1] \
         and logits.shape[2] == att_input.shape[2]:
        with tf.variable_scope("predicted_att"):
          att_matrix = tf.nn.relu(att_input)
          # N C H W
          att_matrix = tf.layers.conv2d(att_matrix,
                                        Nh, 3, padding='same',
                                        data_format='channels_first',
                                        kernel_initializer=kernel_initializer)
          # N C H W
          att_matrix = tf.nn.relu(att_matrix)

          logits = (1 - alpha) * logits + alpha * att_matrix

      self.prev_att_out = logits

      weights = tf.nn.softmax(logits)
      attn_out = tf.matmul(weights, flatten_hw(v, dvh))
      attn_out = tf.reshape(attn_out, [-1, Nh, H, W, dvh])
      attn_out = combine_heads_2d(attn_out)
      with tf.variable_scope("output-proj"):
        attn_out = tf.layers.conv2d(attn_out, dv, 1)

      if do_downsample:
        with tf.variable_scope("aaconv_downsampling"):
          attn_out = tf.image.resize_images(attn_out,
                                            (orgH, orgW),
                                            method=tf.image.ResizeMethod.BILINEAR)

      if self.data_format == 'NCHW':
        attn_out = tf.transpose(attn_out, [0, 3, 1, 2])
        conv = tf.concat([conv_out, attn_out], axis=1)
      else:
        conv = tf.concat([conv_out, attn_out], axis=3)

    if use_batch_norm is None:
      use_batch_norm = self.use_batch_norm
    mlperf.logger.log_conv2d(input_tensor=input_layer, output_tensor=conv,
                             stride_height=d_height, stride_width=d_width,
                             filters=num_out_channels,
                             initializer=kernel_initializer,
                             use_bias=not use_batch_norm and bias is not None)
    if not use_batch_norm:
      if bias is not None:
        biases = self.get_variable('biases', [num_out_channels],
                                   self.variable_dtype, self.dtype,
                                   initializer=tf.constant_initializer(bias))
        biased = tf.reshape(
            tf.nn.bias_add(conv, biases, data_format=self.data_format),
            conv.get_shape())
      else:
        biased = conv
    else:
      self.top_layer = conv
      self.top_size = num_out_channels
      biased = self.batch_norm(**self.batch_norm_config)
    if activation == 'relu':
      mlperf.logger.log(key=mlperf.tags.MODEL_HP_RELU)
      conv1 = tf.nn.relu(biased)
    elif activation == 'linear' or activation is None:
      conv1 = biased
    elif activation == 'tanh':
      conv1 = tf.nn.tanh(biased)
    else:
      raise KeyError('Invalid activation type \'%s\'' % activation)
    self.top_layer = conv1
    self.top_size = num_out_channels
    return conv1

  def conv(self,
           num_out_channels,
           k_height,
           k_width,
           d_height=1,
           d_width=1,
           mode='SAME',
           input_layer=None,
           num_channels_in=None,
           use_batch_norm=None,
           stddev=None,
           activation='relu',
           bias=0.0,
           kernel_initializer=None):
    """Construct a conv2d layer on top of cnn."""
    if input_layer is None:
      input_layer = self.top_layer
    if num_channels_in is None:
      num_channels_in = self.top_size
    if stddev is not None and kernel_initializer is None:
      kernel_initializer = tf.truncated_normal_initializer(stddev=stddev)
    if kernel_initializer is None:
      kernel_initializer = tf.variance_scaling_initializer()
    name = 'conv' + str(self.counts['conv'])
    self.counts['conv'] += 1
    with tf.variable_scope(name):
      strides = [1, d_height, d_width, 1]
      if self.data_format == 'NCHW':
        strides = [strides[0], strides[3], strides[1], strides[2]]
      if mode != 'SAME_RESNET':
        conv = self._conv2d_impl(input_layer, num_channels_in, num_out_channels,
                                 kernel_size=[k_height, k_width],
                                 strides=[d_height, d_width], padding=mode,
                                 kernel_initializer=kernel_initializer)
      else:  # Special padding mode for ResNet models
        if d_height == 1 and d_width == 1:
          conv = self._conv2d_impl(input_layer, num_channels_in,
                                   num_out_channels,
                                   kernel_size=[k_height, k_width],
                                   strides=[d_height, d_width], padding='SAME',
                                   kernel_initializer=kernel_initializer)
        else:
          rate = 1  # Unused (for 'a trous' convolutions)
          kernel_height_effective = k_height + (k_height - 1) * (rate - 1)
          pad_h_beg = (kernel_height_effective - 1) // 2
          pad_h_end = kernel_height_effective - 1 - pad_h_beg
          kernel_width_effective = k_width + (k_width - 1) * (rate - 1)
          pad_w_beg = (kernel_width_effective - 1) // 2
          pad_w_end = kernel_width_effective - 1 - pad_w_beg
          padding = [[0, 0], [pad_h_beg, pad_h_end],
                     [pad_w_beg, pad_w_end], [0, 0]]
          if self.data_format == 'NCHW':
            padding = [padding[0], padding[3], padding[1], padding[2]]
          padded_input_layer = tf.pad(input_layer, padding)
          conv = self._conv2d_impl(padded_input_layer, num_channels_in,
                                   num_out_channels,
                                   kernel_size=[k_height, k_width],
                                   strides=[d_height, d_width], padding='VALID',
                                   kernel_initializer=kernel_initializer)
      if use_batch_norm is None:
        use_batch_norm = self.use_batch_norm
      mlperf.logger.log_conv2d(input_tensor=input_layer, output_tensor=conv,
                               stride_height=d_height, stride_width=d_width,
                               filters=num_out_channels,
                               initializer=kernel_initializer,
                               use_bias=not use_batch_norm and bias is not None)
      if not use_batch_norm:
        if bias is not None:
          biases = self.get_variable('biases', [num_out_channels],
                                     self.variable_dtype, self.dtype,
                                     initializer=tf.constant_initializer(bias))
          biased = tf.reshape(
              tf.nn.bias_add(conv, biases, data_format=self.data_format),
              conv.get_shape())
        else:
          biased = conv
      else:
        self.top_layer = conv
        self.top_size = num_out_channels
        biased = self.batch_norm(**self.batch_norm_config)
      if activation == 'relu':
        mlperf.logger.log(key=mlperf.tags.MODEL_HP_RELU)
        conv1 = tf.nn.relu(biased)
      elif activation == 'linear' or activation is None:
        conv1 = biased
      elif activation == 'tanh':
        conv1 = tf.nn.tanh(biased)
      else:
        raise KeyError('Invalid activation type \'%s\'' % activation)
      self.top_layer = conv1
      self.top_size = num_out_channels
      return conv1

  def _pool(self,
            pool_name,
            pool_function,
            k_height,
            k_width,
            d_height,
            d_width,
            mode,
            input_layer,
            num_channels_in):
    """Construct a pooling layer."""
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = num_channels_in
    name = pool_name + str(self.counts[pool_name])
    self.counts[pool_name] += 1
    if self.use_tf_layers:
      pool = pool_function(
          input_layer, [k_height, k_width], [d_height, d_width],
          padding=mode,
          data_format=self.channel_pos,
          name=name)
    else:
      if self.data_format == 'NHWC':
        ksize = [1, k_height, k_width, 1]
        strides = [1, d_height, d_width, 1]
      else:
        ksize = [1, 1, k_height, k_width]
        strides = [1, 1, d_height, d_width]
      pool = tf.nn.max_pool(input_layer, ksize, strides, padding=mode,
                            data_format=self.data_format, name=name)
    if pool_name == 'mpool':
      mlperf.logger.log_max_pool(input_tensor=input_layer,
                                 output_tensor=pool)
    self.top_layer = pool
    return pool

  def mpool(self,
            k_height,
            k_width,
            d_height=2,
            d_width=2,
            mode='VALID',
            input_layer=None,
            num_channels_in=None):
    """Construct a max pooling layer."""
    return self._pool('mpool', pooling_layers.max_pooling2d, k_height, k_width,
                      d_height, d_width, mode, input_layer, num_channels_in)

  def apool(self,
            k_height,
            k_width,
            d_height=2,
            d_width=2,
            mode='VALID',
            input_layer=None,
            num_channels_in=None):
    """Construct an average pooling layer."""
    return self._pool('apool', pooling_layers.average_pooling2d, k_height,
                      k_width, d_height, d_width, mode, input_layer,
                      num_channels_in)

  def reshape(self, shape, input_layer=None):
    if input_layer is None:
      input_layer = self.top_layer
    self.top_layer = tf.reshape(input_layer, shape)
    self.top_size = shape[-1]  # HACK This may not always work
    return self.top_layer

  def affine(self,
             num_out_channels,
             input_layer=None,
             num_channels_in=None,
             bias=0.0,
             stddev=None,
             activation='relu'):
    if input_layer is None:
      input_layer = self.top_layer
    if num_channels_in is None:
      num_channels_in = self.top_size
    name = 'affine' + str(self.counts['affine'])
    self.counts['affine'] += 1
    with tf.variable_scope(name):
      init_factor = 2. if activation == 'relu' else 1.
      stddev = stddev or np.sqrt(init_factor / num_channels_in)
      kernel = self.get_variable(
          'weights', [num_channels_in, num_out_channels],
          self.variable_dtype, self.dtype,
          initializer=tf.truncated_normal_initializer(stddev=stddev))
      biases = self.get_variable('biases', [num_out_channels],
                                 self.variable_dtype, self.dtype,
                                 initializer=tf.constant_initializer(bias))
      mlperf.logger.log(key=mlperf.tags.MODEL_HP_DENSE,
                        value=num_out_channels)
      logits = tf.nn.xw_plus_b(input_layer, kernel, biases)
      if activation == 'relu':
        mlperf.logger.log(key=mlperf.tags.MODEL_HP_RELU)
        affine1 = tf.nn.relu(logits, name=name)
      elif activation == 'linear' or activation is None:
        affine1 = logits
      else:
        raise KeyError('Invalid activation type \'%s\'' % activation)
      self.top_layer = affine1
      self.top_size = num_out_channels
      return affine1

  def inception_module(self, name, cols, input_layer=None, in_size=None):
    if input_layer is None:
      input_layer = self.top_layer
    if in_size is None:
      in_size = self.top_size
    name += str(self.counts[name])
    self.counts[name] += 1
    with tf.variable_scope(name):
      col_layers = []
      col_layer_sizes = []
      for c, col in enumerate(cols):
        col_layers.append([])
        col_layer_sizes.append([])
        for l, layer in enumerate(col):
          ltype, args = layer[0], layer[1:]
          kwargs = {
              'input_layer': input_layer,
              'num_channels_in': in_size
          } if l == 0 else {}
          if ltype == 'conv':
            self.conv(*args, **kwargs)
          elif ltype == 'mpool':
            self.mpool(*args, **kwargs)
          elif ltype == 'apool':
            self.apool(*args, **kwargs)
          elif ltype == 'share':  # Share matching layer from previous column
            self.top_layer = col_layers[c - 1][l]
            self.top_size = col_layer_sizes[c - 1][l]
          else:
            raise KeyError(
                'Invalid layer type for inception module: \'%s\'' % ltype)
          col_layers[c].append(self.top_layer)
          col_layer_sizes[c].append(self.top_size)
      catdim = 3 if self.data_format == 'NHWC' else 1
      self.top_layer = tf.concat([layers[-1] for layers in col_layers], catdim)
      self.top_size = sum([sizes[-1] for sizes in col_layer_sizes])
      return self.top_layer

  def spatial_mean(self, keep_dims=False):
    name = 'spatial_mean' + str(self.counts['spatial_mean'])
    self.counts['spatial_mean'] += 1
    axes = [1, 2] if self.data_format == 'NHWC' else [2, 3]
    self.top_layer = tf.reduce_mean(
        self.top_layer, axes, keepdims=keep_dims, name=name)
    return self.top_layer

  def dropout(self, keep_prob=0.5, input_layer=None):
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = None
    name = 'dropout' + str(self.counts['dropout'])
    with tf.variable_scope(name):
      if not self.phase_train:
        keep_prob = 1.0
      if self.use_tf_layers:
        dropout = core_layers.dropout(input_layer, 1. - keep_prob,
                                      training=self.phase_train)
      else:
        dropout = tf.nn.dropout(input_layer, keep_prob)
      self.top_layer = dropout
      return dropout

  def _batch_norm_without_layers(self, input_layer, decay, use_scale, epsilon):
    """Batch normalization on `input_layer` without tf.layers."""
    # We make this function as similar as possible to the
    # tf.contrib.layers.batch_norm, to minimize the differences between using
    # layers and not using layers.
    shape = input_layer.shape
    num_channels = shape[3] if self.data_format == 'NHWC' else shape[1]
    beta = self.get_variable('beta', [num_channels], tf.float32, tf.float32,
                             initializer=tf.zeros_initializer())
    if use_scale:
      gamma = self.get_variable('gamma', [num_channels], tf.float32,
                                tf.float32, initializer=tf.ones_initializer())
    else:
      gamma = tf.constant(1.0, tf.float32, [num_channels])
    # For moving variables, we use tf.get_variable instead of self.get_variable,
    # since self.get_variable returns the result of tf.cast which we cannot
    # assign to.
    moving_mean = tf.get_variable('moving_mean', [num_channels],
                                  tf.float32,
                                  initializer=tf.zeros_initializer(),
                                  trainable=False)
    moving_variance = tf.get_variable('moving_variance', [num_channels],
                                      tf.float32,
                                      initializer=tf.ones_initializer(),
                                      trainable=False)
    if self.phase_train:
      bn, batch_mean, batch_variance = tf.nn.fused_batch_norm(
          input_layer, gamma, beta, epsilon=epsilon,
          data_format=self.data_format, is_training=True)
      mean_update = moving_averages.assign_moving_average(
          moving_mean, batch_mean, decay=decay, zero_debias=False)
      variance_update = moving_averages.assign_moving_average(
          moving_variance, batch_variance, decay=decay, zero_debias=False)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mean_update)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, variance_update)
    else:
      bn, _, _ = tf.nn.fused_batch_norm(
          input_layer, gamma, beta, mean=moving_mean,
          variance=moving_variance, epsilon=epsilon,
          data_format=self.data_format, is_training=False)
    return bn

  def batch_norm(self, input_layer=None, decay=0.999, scale=False,
                 epsilon=0.001):
    """Adds a Batch Normalization layer."""
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = None
    name = 'batchnorm' + str(self.counts['batchnorm'])
    self.counts['batchnorm'] += 1

    center = True
    with tf.variable_scope(name) as scope:
      if self.use_tf_layers:
        layer_obj = normalization_layers.BatchNormalization(
            momentum=decay,
            scale=scale,
            epsilon=epsilon,
            fused=True,
            axis=_data_format_to_channel_axis[self.data_format],
            # We pass this 'scope' argument for compatibility with checkpoints
            # created with the contrib version of batch norm. tf_cnn_benchmarks
            # used to use the contrib version.
            _scope=scope,
            center=center,
            name=scope.name)
        bn = layer_obj.apply(input_layer, training=self.phase_train)
      else:
        bn = self._batch_norm_without_layers(input_layer, decay, scale, epsilon)
    self.top_layer = bn
    self.top_size = bn.shape[3] if self.data_format == 'NHWC' else bn.shape[1]
    self.top_size = int(self.top_size)
    mlperf.logger.log_batch_norm(
        input_tensor=input_layer, output_tensor=bn, momentum=decay,
        epsilon=epsilon, center=center, scale=scale, training=self.phase_train)
    return bn

  def lrn(self, depth_radius, bias, alpha, beta):
    """Adds a local response normalization layer."""
    name = 'lrn' + str(self.counts['lrn'])
    self.counts['lrn'] += 1
    self.top_layer = tf.nn.lrn(
        self.top_layer, depth_radius, bias, alpha, beta, name=name)
    return self.top_layer
