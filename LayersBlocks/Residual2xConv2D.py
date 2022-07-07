# Residual block with 2x Conv2D
# https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow_addons.layers import InstanceNormalization

from .FullPreActivation import FullPreActivation

"""
    # Si es necesario, solo cambio cantidad de filtros para ADD, no uso activation en residual
    input_filters = input.shape[-1]
    if num_out_filters != input_filters:
        residual_input = Conv2D(num_out_filters, (1, 1), padding="same", kernel_regularizer=l2(l2_value),
            bias_regularizer=l2(l2_value), name = name_prefix + "--RESIDUAL_INPUT_CONV2D")(input)
    else:
        residual_input = input

    # camino normal (CN)
    x = input
    x = Conv2D(num_out_filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate,
            padding=padding, activation=activation,
            kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value),
            name=name_prefix+"--RESIDUAL_CONV2D_1")(x)
    x = Conv2D(num_out_filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate,
            padding=padding, activation=activation,
            kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value),
            name=name_prefix+"--RESIDUAL_CONV2D_2")(x)

    output = Add(name=name_prefix+"--RESIDUAL_ADD")([residual_input, x])
    return output
"""

@tf.keras.utils.register_keras_serializable()
class Residual2xConv2D(tf.keras.layers.Layer):

    def __init__(self, num_out_filters, name_prefix=None,
                 kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1),
                 padding="same", activation="LR010",  # LR010=LeakyReLU(0.10), RELU=ReLU, None
                 normalization="IN",  # IN=InstanceNormalization, BN=BatchNormalization, None
                 l2_value=0.001, **kwargs):

        super().__init__(name=str(name_prefix)+"_Residual2xConv2D", **kwargs)
        self.num_out_filters = num_out_filters
        self.name_prefix = name_prefix
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.l2_value = l2_value

        if activation == "LR010":
            self.f_activation = LeakyReLU(0.10)
        elif activation == "RELU":
            self.f_activation = ReLU()
        else:
            self.f_activation = None

        self.f_conv2d_1 = Conv2D(filters=num_out_filters, kernel_size=kernel_size, strides=strides,
                                 dilation_rate=dilation_rate, padding=padding,
                                 kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))

        self.f_conv2d_2 = Conv2D(filters=num_out_filters, kernel_size=kernel_size, strides=strides,
                                 dilation_rate=dilation_rate, padding=padding,
                                 kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))

        self.f_add = Add()

    def build(self, input_shape):

        # Si es necesario ajusto la cantidad de filtros finales para poder hacer el ADD
        self.input_channels = input_shape[-1]
        if self.num_out_filters != self.input_channels:
            self.f_conv2d_num_filters = Conv2D(filters=num_out_filters, kernel_size=kernel_size,
                                               strides=strides, dilation_rate=dilation_rate, padding=padding,
                                               kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))

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
        config["kernel_size"] = self.kernel_size
        config["strides"] = self.strides
        config["dilation_rate"] = self.dilation_rate
        config["padding"] = self.padding
        config["activation"] = self.activation,
        config["l2_value"] = self.l2_value
        return config
