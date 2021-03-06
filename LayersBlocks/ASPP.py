# Atrous Spatial Pyramid Pooling Module
# Simil to "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
# Atrous Convolution, and Fully Connected CRFs"
# Use shared FullPreActivation
# https://arxiv.org/pdf/1606.00915v2.pdf

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow_addons.layers import InstanceNormalization

from .FullPreActivation import FullPreActivation

@tf.keras.utils.register_keras_serializable()
class ASPP(tf.keras.layers.Layer):

    def __init__(self, num_out_filters, aspp_rates=[2, 4, 8],
                 kernel_size=(3, 3), strides=(1, 1), padding="same",
                 activation="LR010",  # LR010=LeakyReLU(0.10), RELU=ReLU, None
                 normalization="IN",  # IN=InstanceNormalization, BN=BatchNormalization, None
                 l2_value=0.001, **kwargs):

        super().__init__(**kwargs)
        self.num_out_filters = num_out_filters
        self.aspp_rates = aspp_rates
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation,
        self.normalization = normalization
        self.l2_value = l2_value

        if normalization == "IN":
            self.f_normalization = InstanceNormalization(axis=-1, center=True, scale=True,
                                                         beta_initializer="random_uniform",
                                                         gamma_initializer="random_uniform")
        elif normalization == "BN":
            self.f_normalization = BatchNormalization()
        else:
            self.f_normalization = None

        if activation == "LR010":
            self.f_activation = LeakyReLU(0.10)
        elif activation == "RELU":
            self.f_activation = ReLU()
        else:
            self.f_activation = None

        self.f_conv2d_by_rate = {}

        for rate in self.aspp_rates:
            self.f_conv2d_by_rate[rate] = Conv2D(filters=self.num_out_filters, kernel_size=self.kernel_size,
                                                 strides=self.strides, dilation_rate=rate,
                                                 padding=self.padding, kernel_regularizer=l2(self.l2_value),
                                                 bias_regularizer=l2(self.l2_value))

        self.f_add = Add()

    def call(self, X):

        Y = self.f_normalization(X)
        Y = self.f_activation(Y)

        aspp_operations_by_rate = list()

        for rate in self.aspp_rates:
            aspp_operations_by_rate.append(self.f_conv2d_by_rate[rate](Y))

        Y = self.f_add(aspp_operations_by_rate)

        return Y

    def get_config(self):

        config = super().get_config()
        config["num_out_filters"] = self.num_out_filters
        config["aspp_rates"] = self.aspp_rates
        config["kernel_size"] = self.kernel_size
        config["strides"] = self.strides
        config["padding"] = self.padding
        config["activation"] = self.activation
        config["normalization"] = self.normalization
        config["l2_value"] = self.l2_value
        return config
