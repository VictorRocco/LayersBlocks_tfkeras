# ===================================================
# === Conv2D, optionally with alternative padding ===
# === (reflect, symmetric, constant) ================
# ===================================================

# sources
# - A guide to convolution arithmetic for deep learning
#   https://stats.stackexchange.com/questions/297678/how-to-calculate-optimal-zero-padding-for-convolutional-neural-networks

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2

@tf.keras.utils.register_keras_serializable()
class lbConv2D(tf.keras.layers.Layer):

    def __init__(self, num_out_filters,
                 kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1),
                 padding="reflect", # same, valid, reflect, symmetric
                 activation="LR010",  # LR010=LeakyReLU(0.10), RELU=ReLU, None
                 l2_value=0.001,
                 **kwargs):

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
            output_h = tf.cast(tf.math.ceil(input_h / stride_h), tf.int32)
            output_w = tf.cast(tf.math.ceil(input_w / stride_w), tf.int32)

            if input_h % stride_h == 0:
                pad_along_height = tf.math.maximum((kernel_h - stride_h), 0)
            else:
                pad_along_height = tf.math.maximum(kernel_h - (input_h % stride_h), 0)
            if input_w % stride_w == 0:
                pad_along_width = tf.math.maximum((kernel_w - stride_w), 0)
            else:
                pad_along_width = tf.math.maximum(kernel_w - (input_w % stride_w), 0)

            pad_top = pad_along_height // 2  # amount padding on the top
            pad_bottom = pad_along_height - pad_top  # amount padding on the bottom
            pad_left = pad_along_width // 2  # amount of padding on the left
            pad_right = pad_along_width - pad_left  # amount of padding on the right

            self.tfpad_paddings =  tf.constant([[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
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
