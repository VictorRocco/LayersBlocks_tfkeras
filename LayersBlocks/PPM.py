"""
==============================
=== Pyramid Pooling Module ===
==============================

Simil to Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017
Uses Add instead of Concatenate 1/N filters, and kernel size 3x3 instead of 1x1

https://arxiv.org/abs/1612.01105
https://github.com/hszhao/PSPNet
https://github.com/hszhao/semseg/blob/master/model/pspnet.py

TODO: add residual_add output_mode option
"""

import tensorflow as tf
from tensorflow.keras.layers import Add, AveragePooling2D, Concatenate, UpSampling2D

from .StdCNA import StdCNA


@tf.keras.utils.register_keras_serializable()
class PPM(tf.keras.layers.Layer):
    def __init__(
        self,
        num_out_filters,
        ppm_rates=[2, 4, 8],
        kernel_size=(3, 3),
        strides=(1, 1),
        dilation_rate=(1, 1),
        padding="symmetric",  # same, valid, symmetric, reflect
        activation="LR010",  # LR010=LeakyReLU(0.10), RELU=ReLU, None
        normalization="IN",  # IN=InstanceNormalization, BN=BatchNormalization, None
        output_mode="as_list",  # as_list / add / concatenate
        l2_value=None,
        **kwargs
    ):

        # assert padding: checked on StdCNA
        # assert activation: checked on StdCNA
        # assert normalization: checked on StdCNA
        if output_mode not in ("as_list", "add", "concatenate"):
            raise ValueError("invalid argument: output_mode = ", output_mode)

        super().__init__(**kwargs)
        self.num_out_filters = num_out_filters
        self.ppm_rates = ppm_rates
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activation
        self.normalization = normalization
        self.output_mode = output_mode
        self.l2_value = l2_value

        self.f_avg_pool_2d = {}
        self.f_fnc = {}
        self.f_upsample = {}

        for rate in self.ppm_rates:
            self.f_avg_pool_2d[rate] = AveragePooling2D(pool_size=(rate, rate))
            self.f_fnc[rate] = StdCNA(
                num_out_filters=self.num_out_filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                dilation_rate=self.dilation_rate,
                padding=self.padding,
                activation=self.activation,
                normalization=self.normalization,
                l2_value=self.l2_value,
            )
            self.f_upsample[rate] = UpSampling2D(
                size=(rate, rate), interpolation="bilinear"
            )

        if self.output_mode == "add":
            self.f_final_operation = Add()
        elif self.output_mode == "concatenate":
            self.f_final_operation = Concatenate()
        elif self.output_mode == "as_list":
            self.f_final_operation = None
        else:
            raise ValueError(
                'output_mode should be "as_list", "add" or "concatenate", received: '
                + str(self.output_mode)
                + "."
            )

    def call(self, X):

        ppm_operations_by_rate = []
        for rate in self.ppm_rates:
            Y = self.f_avg_pool_2d[rate](X)
            Y = self.f_fnc[rate](Y)
            Y = self.f_upsample[rate](Y)
            ppm_operations_by_rate.append(Y)
        if self.f_final_operation is not None:
            Y = self.f_final_operation(ppm_operations_by_rate)
        else:
            Y = ppm_operations_by_rate

        return Y

    def get_config(self):

        config = super().get_config()
        config["num_out_filters"] = self.num_out_filters
        config["ppm_rates"] = self.ppm_rates
        config["kernel_size"] = self.kernel_size
        config["strides"] = self.strides
        config["dilation_rate"] = self.dilation_rate
        config["padding"] = self.padding
        config["activation"] = self.activation
        config["normalization"] = self.normalization
        config["output_mode"] = self.output_mode
        config["l2_value"] = self.l2_value
        return config
