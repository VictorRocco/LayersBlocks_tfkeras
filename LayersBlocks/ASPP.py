# =============================================
# === Atrous Spatial Pyramid Pooling Module ===
# =============================================
# Simil to "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
# Atrous Convolution, and Fully Connected CRFs"
# Use shared FullPreActivation
# https://arxiv.org/pdf/1606.00915v2.pdf

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow_addons.layers import InstanceNormalization

from .Normalization import Normalization
from .Activation import Activation
from .lbConv2D import lbConv2D

@tf.keras.utils.register_keras_serializable()
class ASPP(tf.keras.layers.Layer):

    def __init__(self, num_out_filters, aspp_rates=[2, 4, 8],
                 kernel_size=(3, 3), strides=(1, 1), padding="symmetric",
                 activation="LR010",  # LR010=LeakyReLU(0.10), RELU=ReLU, None
                 normalization="IN",  # IN=InstanceNormalization, BN=BatchNormalization, None
                 l2_value=0.001, **kwargs):

        super().__init__(**kwargs)
        self.num_out_filters = num_out_filters
        self.aspp_rates = aspp_rates
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.normalization = normalization
        self.l2_value = l2_value

        self.f_normalization = Normalization(normalization=self.normalization)
        self.f_activation = Activation(activation=self.activation)

        self.f_conv2d_by_rate = {}
        for rate in self.aspp_rates:
            self.f_conv2d_by_rate[rate] = lbConv2D(num_out_filters=self.num_out_filters, kernel_size=self.kernel_size,
                                                   strides=self.strides, dilation_rate=(rate, rate),
                                                   padding=self.padding, activation=None, l2_value=self.l2_value)

        self.f_add = Add()

    def call(self, X):

        Y = X
        Y = self.f_normalization(Y)
        Y = self.f_activation(Y)

        aspp_operations_by_rate = list()
        for rate in self.aspp_rates:
            aspp_operation = self.f_conv2d_by_rate[rate](Y)
            aspp_operations_by_rate.append(aspp_operation)
            # print("in", X.shape, "out", aspp_operation.shape, "kernel", self.kernel_size, "strides", self.strides,
            #      "rate", rate, flush=True)
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
