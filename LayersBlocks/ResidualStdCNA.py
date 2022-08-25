# Residual block using StdCNA.
# https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow_addons.layers import InstanceNormalization

from .StdCNA import StdCNA

@tf.keras.utils.register_keras_serializable()
class ResidualStdCNA(tf.keras.layers.Layer):

    def __init__(self, num_out_filters,
                 kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1),
                 padding="symmetric", # same, valid, symetric, reflect
                 activation="LR010",  # LR010=LeakyReLU(0.10), RELU=ReLU, None
                 normalization="IN",  # IN=InstanceNormalization, BN=BatchNormalization, None
                 l2_value=0.001, **kwargs):

        super().__init__(**kwargs)
        self.num_out_filters = num_out_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activation,
        self.normalization = normalization
        self.l2_value = l2_value

        self.f_fpa_1 = StdCNA(num_out_filters=self.num_out_filters,
                              kernel_size=self.kernel_size, strides=self.strides,
                              dilation_rate=self.dilation_rate,
                              padding=self.padding, activation=self.activation,
                              normalization=self.normalization, l2_value=self.l2_value)

        self.f_fpa_2 = StdCNA(num_out_filters=self.num_out_filters,
                              kernel_size=self.kernel_size, strides=self.strides,
                              dilation_rate=self.dilation_rate,
                              padding=self.padding, activation=self.activation,
                              normalization=self.normalization, l2_value=self.l2_value)

        self.f_add = Add()

    def build(self, input_shape):

        # Si es necesario ajusto la cantidad de filtros finales para poder hacer el Residual ADD
        self.input_channels = input_shape[-1]
        if self.num_out_filters != self.input_channels:
            self.f_conv2d_num_filters = Conv2D(filters=self.num_out_filters, kernel_size=(1, 1))

    def call(self, X):

        Y = X

        Y = self.f_fpa_1(Y)
        Y = self.f_fpa_2(Y)

        # Si es necesario ajusto la cantidad de filtros finales para poder hacer el Residual ADD
        if self.num_out_filters != self.input_channels:
            Y = self.f_add([self.f_conv2d_num_filters(X), Y])
        else:
            Y = self.f_add([X, Y])

        return Y

    def get_config(self):

        config = super().get_config()
        config["num_out_filters"] = self.num_out_filters
        config["kernel_size"] = self.kernel_size
        config["strides"] = self.strides
        config["dilation_rate"] = self.dilation_rate
        config["padding"] = self.padding
        config["activation"] = self.activation
        config["normalization"] = self.normalization
        config["l2_value"] = self.l2_value
        return config
