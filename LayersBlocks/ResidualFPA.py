# Residual block using Full Pre Activation.
# https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow_addons.layers import InstanceNormalization

from .FullPreActivation import FullPreActivation

@tf.keras.utils.register_keras_serializable()
class ResidualFPA(tf.keras.layers.Layer):

    def __init__(self, num_out_filters, name_prefix=None,
                 kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1),
                 padding="same", activation="LR010",  # LR010=LeakyReLU(0.10), RELU=ReLU, None
                 normalization="IN",  # IN=InstanceNormalization, BN=BatchNormalization, None
                 l2_value=0.001, **kwargs):

        super().__init__(name=str(name_prefix)+"_ResFPA", **kwargs)
        self.num_out_filters = num_out_filters
        self.name_prefix = name_prefix
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activation,
        self.normalization = normalization
        self.l2_value = l2_value

        self.f_fpa_1 = FullPreActivation(num_out_filters=self.num_out_filters, name_prefix=None,
                                         kernel_size=self.kernel_size, strides=self.strides,
                                         dilation_rate=self.dilation_rate,
                                         padding=self.padding, activation=self.activation,
                                         normalization=self.normalization, l2_value=self.l2_value)

        self.f_fpa_2 = FullPreActivation(num_out_filters=self.num_out_filters, name_prefix=None,
                                         kernel_size=self.kernel_size, strides=self.strides,
                                         dilation_rate=self.dilation_rate,
                                         padding=self.padding, activation=self.activation,
                                         normalization=self.normalization, l2_value=self.l2_value)

        # Si es necesario ajusto la cantidad de filtros finales para poder hacer el ADD
        self.f_conv2d = Conv2D(filters=num_out_filters, kernel_size=(1, 1), padding="same", kernel_regularizer=l2(l2_value),
                               bias_regularizer=l2(l2_value))

        self.f_add = Add()

    def call(self, X):

        Y = self.f_fpa_1(X)
        Y = self.f_fpa_2(Y)

        # Si es necesario ajusto la cantidad de filtros finales para poder hacer el ADD
        input_filters = X.shape[-1]
        if self.num_out_filters != input_filters:
            Y = self.f_add([self.f_conv2d(X), Y])
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
        config["normalization"] = self.normalization
        config["l2_value"] = self.l2_value
        return config
