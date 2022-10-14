# ===================================================
# === Conv2D, optionally with alternative padding ===
# === (reflect, symmetric, constant) ================
# ===================================================

# sources
# - How to keep the shape of input and output same when dilation conv?
#   https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
# - Implementing 'SAME' and 'VALID' padding of Tensorflow in Python
#   https://mmuratarat.github.io/2019-01-17/implementing-padding-schemes-of-tensorflow-in-python
# - Convolution Visualizer
#   https://ezyang.github.io/convolution-visualizer/index.html

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2

from .Activation import Activation

@tf.keras.utils.register_keras_serializable()
class lbConv2D(tf.keras.layers.Layer):

    def __init__(self, num_out_filters,
                 kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1),
                 padding="symmetric", # same, valid, symmetric, reflect
                 activation="LR010",  # LR010=LeakyReLU(0.10), RELU=ReLU, None
                 l2_value=None,
                 **kwargs):

        assert padding == "same" or \
               padding == "valid" or \
               padding == "symmetric" or \
               padding == "reflect", \
            "padding parameter not valid"
        # assert activation: checked on Activation
        # assert normalization: checked on Normalization

        super().__init__(**kwargs)
        self.num_out_filters = num_out_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activation
        self.l2_value = l2_value

        self.f_activation = Activation(activation=self.activation)

    def build(self, input_shape):
        if (self.padding == "same") or (self.padding == "valid"):
            # use standard Conv2d
            self.f_conv2d = Conv2D(filters=self.num_out_filters, kernel_size=self.kernel_size,
                                   strides=self.strides, dilation_rate=self.dilation_rate,
                                   padding=self.padding, kernel_regularizer=l2(self.l2_value),
                                   bias_regularizer=l2(self.l2_value))
        else:
            # custom padding (reflect or symmetric)
            input_h = input_shape[1]
            input_w = input_shape[2]
            stride_h = self.strides[0]
            stride_w = self.strides[1]
            kernel_h = self.kernel_size[0]
            kernel_w = self.kernel_size[1]
            dilation_h = self.dilation_rate[0]
            dilation_w = self.dilation_rate[1]

            # NOTE: "original" works when dilation = 1, if dilatio != 1 it does'nt work
            # I fixed dilation != 1 with:
            # 1) https://mmuratarat.github.io/2019-01-17/implementing-padding-schemes-of-tensorflow-in-python
            # 2) https://ezyang.github.io/convolution-visualizer/index.html
            # The visualizer shows padding = 2 * dilation when: (input = N * 8) and Kernel = 3 (odd)
            # Also the correct formulae is )not implemented here, but verified with convolution visualizer):
            # padding (each side) = [(input-1)*stride -input +kernel + (kernel-1)*(dil-1)] / 2
            if (kernel_h == 1 and stride_h == 1 and dilation_h == 1):
                pad_along_height = 0
            elif (input_h % stride_h == 0):
                # pad_along_height = tf.math.maximum((kernel_h - stride_h), 0) #original
                pad_along_height = tf.math.maximum((kernel_h - stride_h), 2 * dilation_h) #VNR
            else:
                # pad_along_height = tf.math.maximum(kernel_h - (input_h % stride_h), 0) #original
                pad_along_height = tf.math.maximum(kernel_h - (input_h % stride_h), 2 * dilation_h) #VNR
            if (kernel_w == 1 and stride_w == 1 and dilation_w == 1):
                pad_along_width = 0
            elif (input_w % stride_w == 0):
                # pad_along_width = tf.math.maximum((kernel_w - stride_w), 0) #original
                pad_along_width = tf.math.maximum((kernel_w - stride_w), 2 * dilation_w) #VNR
            else:
                pad_along_width = tf.math.maximum(kernel_w - (input_w % stride_w), 0) #original
                # pad_along_width = tf.math.maximum(kernel_w - (input_w % stride_w), 2 * dilation_w) #VNR

            pad_top = int(pad_along_height // 2)  # amount padding on the top
            pad_bottom = int(pad_along_height - pad_top)  # amount padding on the bottom
            pad_left = int(pad_along_width // 2)  # amount of padding on the left
            pad_right = int(pad_along_width - pad_left)  # amount of padding on the right

            # print("pad_top", pad_top, "pad_bottom", pad_bottom, "pad_left", pad_left, "pad_right", pad_right)

            self.tfpad_paddings =  tf.constant([[0, 0], [pad_top, pad_bottom],
                                                [pad_left, pad_right], [0, 0]])
            self.f_conv2d = Conv2D(filters=self.num_out_filters, kernel_size=self.kernel_size,
                                   strides=self.strides, dilation_rate=self.dilation_rate,
                                   padding="valid", kernel_regularizer=l2(self.l2_value),
                                   bias_regularizer=l2(self.l2_value))

    def call(self, X):

        Y = X
        if (self.padding == "same") or (self.padding == "valid"):
            # use standard Conv2d
            Y = self.f_conv2d(Y)
            Y = self.f_activation(Y)
        else:
            # custom padding (reflect or symmetric)
            Y = tf.pad(Y, self.tfpad_paddings, self.padding)
            Y = self.f_conv2d(Y)
            Y = self.f_activation(Y)
        return Y

    def get_config(self):

        config = super().get_config()
        config["num_out_filters"] = self.num_out_filters
        config["kernel_size"] = self.kernel_size
        config["strides"] = self.strides
        config["dilation_rate"] = self.dilation_rate
        config["padding"] = self.padding
        config["activation"] = self.activation
        config["l2_value"] = self.l2_value
        return config
