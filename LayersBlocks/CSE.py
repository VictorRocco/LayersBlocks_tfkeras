"""
=============================================
=== Channel-wise Squeeze and Excite block ===
=============================================

https://arxiv.org/abs/1803.02579 - official paper
"Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks"

https://arxiv.org/abs/1709.01507 - official paper "Squeeze and Excitation Networks"
https://github.com/hujie-frank/SENet - official implementation "Squeeze and Excitation Networks"
https://github.com/titu1994/keras-squeeze-excite-network/blob/master/se.py
https://github.com/RayXie29/SENet_Keras
https://blog.paperspace.com/channel-attention-squeeze-and-excitation-networks/
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Multiply
from tensorflow.keras.regularizers import l2

from .Activation import Activation
from .lbConv2D import lbConv2D


@tf.keras.utils.register_keras_serializable()
class CSE(tf.keras.layers.Layer):

    def __init__(
            self,
            activation="LR010",  # LR010=LeakyReLU(0.10), RELU=ReLU, None
            num_out_filters=None,  # None, num_out_filters
            l2_value=None,
            ratio=16,
            **kwargs):

        # assert activation: checked on Activation

        super().__init__(**kwargs)
        self.activation = activation
        self.num_out_filters = num_out_filters
        self.l2_value = l2_value
        self.ratio = ratio

        self.f_activation = Activation(activation=self.activation)
        self.f_gap = GlobalAveragePooling2D()
        self.f_multiply = Multiply()

    def build(self, input_shape):
        self.input_channels = input_shape[-1]
        # Configuration
        self.reduced_channels = self.input_channels // self.ratio
        if self.reduced_channels <= 1:
            self.reduced_channels = 2

        self.f_dense_1 = Dense(
            self.reduced_channels,
            kernel_initializer="he_normal",
            use_bias=False,
            activation=self.f_activation,
            kernel_regularizer=l2(self.l2_value),
            bias_regularizer=l2(self.l2_value),
        )
        self.f_dense_2 = Dense(
            self.input_channels,
            kernel_initializer="he_normal",
            use_bias=False,
            activation="sigmoid",
            kernel_regularizer=l2(self.l2_value),
            bias_regularizer=l2(self.l2_value),
        )

        if (self.num_out_filters is not None) and (self.num_out_filters != self.input_channels):
            self.f_matching_num_filters = lbConv2D(
                num_out_filters=self.num_out_filters,
                kernel_size=(1, 1),
                activation=None,
                l2_value=self.l2_value,
            )  # NOTE: no activation

    def call(self, X):

        # Squeeze operation
        Y = self.f_gap(X)
        Y = self.f_dense_1(Y)

        # Excitation operation
        Y = self.f_dense_2(Y)

        Y = self.f_multiply([X, Y])

        # Matching filters
        if (self.num_out_filters is not None) and (self.num_out_filters != self.input_channels):
            Y = self.f_matching_num_filters(Y)

        return Y

    def get_config(self):

        config = super().get_config()
        config["activation"] = self.activation
        config["num_out_filters"] = self.num_out_filters
        config["l2_value"] = self.l2_value
        config["ratio"] = self.ratio
        return config
