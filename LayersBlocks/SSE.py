"""
=============================================
=== Spatial-wise Squeeze and Excite block ===
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
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2


@tf.keras.utils.register_keras_serializable()
class SSE(tf.keras.layers.Layer):
    def __init__(self, l2_value=None, **kwargs):

        super().__init__(**kwargs)
        self.l2_value = l2_value

        self.f_sse = Conv2D(
            1,
            (1, 1),
            activation="sigmoid",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=l2(self.l2_value),
            bias_regularizer=l2(self.l2_value),
        )

        self.f_multiply = Multiply()

    def call(self, X):

        # Squeeze operation
        Y = self.f_sse(X)

        # Excitation operation
        Y = self.f_multiply([X, Y])

        return Y

    def get_config(self):

        config = super().get_config()
        config["l2_value"] = self.l2_value
        return config
