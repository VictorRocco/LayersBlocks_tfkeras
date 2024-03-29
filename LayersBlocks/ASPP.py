"""
=============================================
=== Atrous Spatial Pyramid Pooling Module ===
=============================================

Simil to "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
Atrous Convolution, and Fully Connected CRFs"

https://arxiv.org/pdf/1606.00915v2.pdf
"""

import tensorflow as tf
from tensorflow.keras.layers import Add, Concatenate, Conv2D

from .StdCNA import StdCNA


@tf.keras.utils.register_keras_serializable()
class ASPP(tf.keras.layers.Layer):

    def __init__(
            self,
            num_out_filters,
            aspp_rates=[2, 4, 8],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="symmetric",  # same, valid, symmetric, reflect
            activation="LR010",  # LR010=LeakyReLU(0.10), RELU=ReLU, None
            normalization="IN",  # IN=InstanceNormalization, BN=BatchNormalization, None
            output_mode="as_list",  # as_list / residual_add / add / concatenate
            l2_value=None,
            **kwargs):

        # assert padding: checked on StdCNA
        # assert activation: checked on StdCNA
        # assert normalization: checked on StdCNA
        if output_mode not in ("as_list", "residual_add", "add", "concatenate"):
            raise ValueError("invalid argument: output_mode = ", output_mode)

        super().__init__(**kwargs)
        self.num_out_filters = num_out_filters
        self.aspp_rates = aspp_rates
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.normalization = normalization
        self.output_mode = output_mode
        self.l2_value = l2_value

        self.f_stdcna_by_rate = {}
        for rate in self.aspp_rates:
            self.f_stdcna_by_rate[rate] = StdCNA(
                num_out_filters=self.num_out_filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                dilation_rate=(rate, rate),
                padding=self.padding,
                activation=self.activation,
                l2_value=self.l2_value,
            )

        if self.output_mode == "residual_add":
            self.f_final_operation = Add()
        elif self.output_mode == "add":
            self.f_final_operation = Add()
        elif self.output_mode == "concatenate":
            self.f_final_operation = Concatenate()
        elif self.output_mode == "as_list":
            self.f_final_operation = None
        else:
            raise ValueError(
                'output_mode should be "as_list", "residual_add", "add" or "concatenate", received: ' +
                str(self.output_mode) + ".")

    def build(self, input_shape):

        # Si es necesario ajusto la cantidad de filtros finales
        # para poder hacer el Residual ADD (sin activacion)
        self.input_channels = input_shape[-1]
        if self.num_out_filters != self.input_channels:
            self.f_output_conv2d_num_filters = Conv2D(filters=self.num_out_filters, kernel_size=(1, 1))

    def call(self, X):

        Y = X
        aspp_operations_by_rate = []
        for rate in self.aspp_rates:
            aspp_operation = self.f_stdcna_by_rate[rate](Y)
            aspp_operations_by_rate.append(aspp_operation)
            # print("in", X.shape, "out", aspp_operation.shape, "kernel", self.kernel_size,
            # "strides", self.strides, "rate", rate, flush=True)

        if self.f_final_operation == "residual_add":
            # Si es necesario ajusto la cantidad de filtros finales para poder hacer el Residual ADD
            if self.num_out_filters != self.input_channels:
                aspp_operations_by_rate.append(self.f_output_conv2d_num_filters(X))
            else:
                aspp_operations_by_rate.append(X)

        if self.f_final_operation is not None:
            Y = self.f_final_operation(aspp_operations_by_rate)
        else:
            Y = aspp_operations_by_rate

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
        config["output_mode"] = self.output_mode
        config["l2_value"] = self.l2_value
        return config
