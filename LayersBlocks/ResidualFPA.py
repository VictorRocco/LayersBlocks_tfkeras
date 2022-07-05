# Residual block using Full Pre Activation.
# https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow_addons.layers import InstanceNormalization

from .FullPreActivation import FullPreActivation

@tf.keras.utils.register_keras_serializable()
class ResidualFPA(tf.keras.layers.Layer):

    def __init__(self, num_out_filters, name_prefix,
                 kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1),
                 padding="same", activation="LR010",  # LR010=LeakyReLU(0.10), RELU=ReLU, None
                 normalization="IN",  # IN=InstanceNormalization, BN=BatchNormalization, None
                 l2_value=0.001, **kwargs):

        super().__init__(**kwargs)
        self.set_config(num_out_filters, name_prefix,
                        kernel_size, strides, dilation_rate,
                        padding, activation,
                        normalization, l2_value)

        self.f_fpa_1 = FullPreActivation(num_out_filters, name_prefix+"_FPA1",
                                         kernel_size, strides, dilation_rate,
                                         padding, activation,
                                         normalization, l2_value)

        self.f_fpa_2 = FullPreActivation(num_out_filters, name_prefix+"_FPA2",
                                         kernel_size, strides, dilation_rate,
                                         padding, activation,
                                         normalization, l2_value)

        # Si es necesario ajusto la cantidad de filtros finales para poder hacer el ADD
        self.f_conv2d = Conv2D(num_out_filters, (1, 1), padding="same", kernel_regularizer=l2(l2_value),
                               bias_regularizer=l2(l2_value), name=name_prefix+"_CONV2D")

    def call(self, X):

        Y = self.f_fpa_1(X)
        Y = self.f_fpa_2(Y)

        # Si es necesario ajusto la cantidad de filtros finales para poder hacer el ADD
        input_filters = X.shape[-1]
        if self.num_out_filters != input_filters:
            Y = Add(name=self.name_prefix+"_ADD")([self.f_conv2d(X), Y])
        else:
            Y = Add(name=self.name_prefix+"_ADD")([X, Y])

        return Y

    def set_config(self, num_out_filters, name_prefix,
                   kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1),
                   padding="same", activation="LR010",  # LR010=LeakyReLU(0.10), RELU=ReLU, None
                   normalization="IN",  # IN=InstanceNormalization, BN=BatchNormalization, None
                   l2_value=0.001, **kwargs):

        self.num_out_filters = num_out_filters
        self.name_prefix = name_prefix
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activation,
        self.normalization = normalization
        self.l2_value = l2_value

    def get_config(self):

        config = super().get_config()
        config["num_out_filters"] = self.num_out_filters
        config["name_prefix"] = self.name_prefix
        config["kernel_size"] = self.kernel_size
        config["strides"] = self.strides
        config["dilation_rate"] = self.dilation_rate
        config["padding"] = self.padding
        config["activation"] = self.activation,
        config["normalization"] = self.normalization
        config["l2_value"] = self.l2_value
        return config
