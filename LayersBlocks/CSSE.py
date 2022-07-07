# Concurrent Channel/Spatial-wise Squeeze and Excite block

# https://arxiv.org/abs/1709.01507 - official paper "Squeeze and Excitation Networks"
# https://github.com/hujie-frank/SENet - official implementation "Squeeze and Excitation Networks"
# https://arxiv.org/abs/1803.02579 - official paper
# 		"Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks"
# https://github.com/titu1994/keras-squeeze-excite-network/blob/master/se.py
# https://github.com/RayXie29/SENet_Keras
# https://blog.paperspace.com/channel-attention-squeeze-and-excitation-networks/

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2

from . import CSE, SSE

@tf.keras.utils.register_keras_serializable()
class CSSE(tf.keras.layers.Layer):

	def __init__(self, name_prefix=None, activation="LR010", #LR010=LeakyReLU(0.10), RELU=ReLU, None
				 l2_value=0.001, ratio=16, **kwargs):
        			
		super().__init__(name=str(name_prefix)+"_CSSE", **kwargs)
		self.name_prefix = name_prefix
		self.activation = activation
		self.l2_value = l2_value
		self.ratio = ratio

		if self.activation == "LR010":
			self.f_activation = LeakyReLU(0.10)
		elif self.activation == "RELU":
			self.f_activation = ReLU()
		else:
			self.f_activation = None

		self.f_cse = CSE(name_prefix=None, activation=self.activation, l2_value=self.l2_value, ratio=self.ratio)

		self.f_sse = SSE(name_prefix=None, l2_value=self.l2_value)

		self.f_add = Add()

	def call(self, X):

		Y = self.f_add([self.f_cse(X), self.f_sse(X)])

		return Y

	def get_config(self):

		config = super().get_config()
		config["name_prefix"] = self.name_prefix
		config["activation"] = self.activation
		config["l2_value"] = self.l2_value
		config["ratio"] = self.ratio
		return config
