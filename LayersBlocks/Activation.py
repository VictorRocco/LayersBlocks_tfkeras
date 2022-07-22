# ========================================
# === selected Activation by parameter ===
# ========================================

import tensorflow as tf

from tf.keras.layers import *
from tf.keras.regularizers import l2

@tf.keras.utils.register_keras_serializable()
class Activation(tf.keras.layers.Layer):

	def __init__(self, activation="LR010", #LR010=LeakyReLU(0.10), RELU=ReLU, None
				 **kwargs):
        			
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
		if self.f_activation != None:
			Y = self.activation(Y)
		return Y

	def get_config(self):

		config = super().get_config()
		config["activation"] = self.activation
		return config