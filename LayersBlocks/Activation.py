# ========================================
# === selected Activation by parameter ===
# ========================================

import tensorflow as tf
from tensorflow.keras.layers import *


@tf.keras.utils.register_keras_serializable()
class Activation(tf.keras.layers.Layer):
    def __init__(
        self, activation="LR010", **kwargs  # LR010=LeakyReLU(0.10), RELU=ReLU, None
    ):

        if activation not in ("LR010", "RELU", None):
            raise ValueError("invalid argument: activation = ", activation)

        super().__init__(**kwargs)
        self.activation = activation

        if self.activation == "LR010":
            self.f_activation = LeakyReLU(0.10)
        elif self.activation == "RELU":
            self.f_activation = ReLU()
        else:
            self.f_activation = None

    def call(self, X):

        Y = X
        if self.f_activation is not None:
            Y = self.f_activation(Y)
        return Y

    def get_config(self):

        config = super().get_config()
        config["activation"] = self.activation
        return config
