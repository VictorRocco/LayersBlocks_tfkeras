# Residual block with 2x Conv2D
# https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2

@tf.keras.utils.register_keras_serializable()
class ResidualConv2D(tf.keras.layers.Layer):

    def __init__(self, num_out_filters, name_prefix=None,
                 kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1),
                 padding="same", activation="LR010",  # LR010=LeakyReLU(0.10), RELU=ReLU, None
                 l2_value=0.001, **kwargs):

        super().__init__(name=str(name_prefix)+"_ResidualConv2D", **kwargs)
        self.num_out_filters = num_out_filters
        self.name_prefix = name_prefix
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activation
        self.l2_value = l2_value

        if activation == "LR010":
            self.f_activation = LeakyReLU(0.10)
        elif activation == "RELU":
            self.f_activation = ReLU()
        else:
            self.f_activation = None

        self.f_conv2d_1 = Conv2D(filters=self.num_out_filters, kernel_size=self.kernel_size,
                                 strides=self.strides, dilation_rate=self.dilation_rate,
                                 padding=self.padding, kernel_regularizer=l2(self.l2_value),
                                 bias_regularizer=l2(self.l2_value))

        self.f_conv2d_2 = Conv2D(filters=self.num_out_filters, kernel_size=self.kernel_size,
                                 strides=self.strides, dilation_rate=self.dilation_rate,
                                 padding=self.padding, kernel_regularizer=l2(self.l2_value),
                                 bias_regularizer=l2(self.l2_value))

        self.f_add = Add()

    def build(self, input_shape):

        # Si es necesario ajusto la cantidad de filtros finales para poder hacer el ADD
        self.input_channels = input_shape[-1]
        if self.num_out_filters != self.input_channels:
            self.f_conv2d_num_filters = Conv2D(filters=self.num_out_filters, kernel_size=self.kernel_size,
                                               strides=self.strides, dilation_rate=self.dilation_rate,
                                               padding=self.padding, kernel_regularizer=l2(self.l2_value),
                                               bias_regularizer=l2(self.l2_value))

    def call(self, X):

        Y = self.f_conv2d_1(X)
        Y = self.f_conv2d_2(Y)

        # Si es necesario ajusto la cantidad de filtros finales para poder hacer el ADD
        if self.num_out_filters != self.input_channels:
            Y = self.f_add([self.f_conv2d_num_filters(X), Y])
        else:
            Y = self.f_add([X, Y])

        return Y

    def get_config(self):

        config = super().get_config()
        config["num_out_filters"] = self.num_out_filters
        config["name_prefix"] = self.name_prefix
        config["kernel_size"] = self.kernel_size
        config["strides"] = self.strides
        config["dilation_rate"] = self.dilation_rate
        config["padding"] = self.padding
        config["activation"] = self.activation
        config["l2_value"] = self.l2_value
        return config
