# ========================================
# === selected Activation by parameter ===
# ========================================

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow_addons.layers import InstanceNormalization


@tf.keras.utils.register_keras_serializable()
class Normalization(tf.keras.layers.Layer):

    def __init__(self, normalization="IN",  # IN=InstanceNormalization, BN=BatchNormalization, None
                 **kwargs):

        if normalization not in ("IN", "BN", None):
            raise ValueError("invalid argument: normalization = ", normalization)

        super().__init__(**kwargs)
        self.normalization = normalization

        if normalization == "IN":
            self.f_normalization = InstanceNormalization(axis=-1, center=True, scale=True,
                                                         beta_initializer="random_uniform",
                                                         gamma_initializer="random_uniform")
        elif normalization == "BN":
            self.f_normalization = BatchNormalization()
        else:
            self.f_normalization = None

    def call(self, X):

        Y = X
        if self.f_normalization != None:
            Y = self.f_normalization(Y)
        return Y

    def get_config(self):

        config = super().get_config()
        config["normalization"] = self.normalization
        return config
