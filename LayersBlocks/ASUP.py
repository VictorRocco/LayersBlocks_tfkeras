# ==========================================
# === Atrous Spatial UNET Pooling Module ===
# ==========================================
# Simil to class RSU4F in U2NET.
# (https://github.com/xuebinqin/U-2-Net/blob/master/model/u2net.py)

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow_addons.layers import InstanceNormalization

from .StdCNA import StdCNA


@tf.keras.utils.register_keras_serializable()
class ASUP(tf.keras.layers.Layer):
    def __init__(
        self,
        num_out_filters,
        asup_rates=[1, 2, 4, 8],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="symmetric",  # same, valid, symmetric, reflect
        activation="LR010",  # LR010=LeakyReLU(0.10), RELU=ReLU, None
        normalization="IN",  # IN=InstanceNormalization, BN=BatchNormalization, None
        l2_value=None,
        **kwargs
    ):

        # assert padding: checked on StdCNA
        # assert activation: checked on StdCNA
        # assert normalization: checked on StdCNA

        super().__init__(**kwargs)
        self.num_out_filters = num_out_filters
        self.asup_rates = asup_rates
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.normalization = normalization
        self.l2_value = l2_value

        self.f_fpa = {}
        self.f_add = {}
        for rate in self.asup_rates:
            self.f_fpa[rate] = StdCNA(
                num_out_filters=self.num_out_filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                dilation_rate=rate,
                padding=self.padding,
                activation=self.activation,
                normalization=self.normalization,
                l2_value=self.l2_value,
            )
            self.f_add[rate] = Add()

    def call(self, X):

        Y = X

        asup_operations_by_rate = {}
        for rate in self.asup_rates:
            Y = self.f_fpa[rate](Y)
            asup_operations_by_rate[rate] = Y

        for i, rate in enumerate(reversed(self.asup_rates)):
            if i == 0:
                Y = asup_operations_by_rate[rate]
            else:
                Y = self.f_add[rate]([Y, asup_operations_by_rate[rate]])

        return Y

    def get_config(self):

        config = super().get_config()
        config["num_out_filters"] = self.num_out_filters
        config["asup_rates"] = self.asup_rates
        config["kernel_size"] = self.kernel_size
        config["strides"] = self.strides
        config["padding"] = self.padding
        config["activation"] = self.activation
        config["normalization"] = self.normalization
        config["l2_value"] = self.l2_value
        return config
