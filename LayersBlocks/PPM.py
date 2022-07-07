# Pyramid Pooling Module
# Simil to Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017
# Uses FullPreActivation, and Add instead of Concatenate 1/N filters
# https://arxiv.org/abs/1612.01105
# https://github.com/hszhao/PSPNet
# https://github.com/hszhao/semseg/blob/master/model/pspnet.py

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow_addons.layers import InstanceNormalization

from .FullPreActivation import FullPreActivation

@tf.keras.utils.register_keras_serializable()
class PPM(tf.keras.layers.Layer):

    def __init__(self, num_out_filters, ppm_rates=[2, 4, 8], name_prefix=None,
                 kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding="same",
                 activation="LR010",  # LR010=LeakyReLU(0.10), RELU=ReLU, None
                 normalization="IN",  # IN=InstanceNormalization, BN=BatchNormalization, None
                 l2_value=0.001, **kwargs):

        super().__init__(name=str(name_prefix)+"_PPM", **kwargs)
        self.num_out_filters = num_out_filters
        self.ppm_rates = ppm_rates
        self.name_prefix = name_prefix
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activation,
        self.normalization = normalization
        self.l2_value = l2_value

        self.f_avg_pool_2d = {}
        self.f_fpa = {}
        self.f_upsample = {}

        for rate in self.ppm_rates:
            self.f_avg_pool_2d[rate] = AveragePooling2D(pool_size=(rate, rate))
            self.f_fpa[rate] = FullPreActivation(num_out_filters=self.num_out_filters,
                                                 name_prefix=self.name_prefix, kernel_size=self.kernel_size,
                                                 strides=self.strides, dilation_rate=self.dilation_rate,
                                                 padding=self.padding, activation=self.activation,
                                                 normalization=self.normalization, l2_value=self.l2_value)
            self.f_upsample[rate] = UpSampling2D(size=(rate, rate), interpolation="bilinear")

        self.f_add = Add()

    def call(self, X):

        ppm_operations_by_rate = list()

        for rate in self.ppm_rates:
            Y = self.f_avg_pool_2d[rate](X)
            Y = self.f_fpa[rate](Y)
            Y = self.f_upsample[rate](Y)
            ppm_operations_by_rate.append(Y)

        Y = self.f_add(ppm_operations_by_rate)

        return Y

    def get_config(self):

        config = super().get_config()
        config["num_out_filters"] = self.num_out_filters
        config["ppm_rates"] = self.ppm_rates
        config["name_prefix"] = self.name_prefix
        config["kernel_size"] = self.kernel_size
        config["strides"] = self.strides
        config["dilation_rate"] = self.dilation_rate
        config["padding"] = self.padding
        config["activation"] = self.activation
        config["normalization"] = self.normalization
        config["l2_value"] = self.l2_value
        return config