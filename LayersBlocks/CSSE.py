# Concurrent Channel/Spatial-wise Squeeze and Excite block

#TODO: use Activation

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

	def __init__(self, activation="LR010",  #LR010=LeakyReLU(0.10), RELU=ReLU, None
				 num_out_filters=None,  # None, num_out_filters
				 l2_value=None, ratio=16, **kwargs):
        			
		super().__init__(**kwargs)
		self.activation = activation
		self.num_out_filters = num_out_filters
		self.l2_value = l2_value
		self.ratio = ratio

		self.f_cse = CSE(activation=self.activation, num_out_filters=self.num_out_filters,
						 l2_value=self.l2_value, ratio=self.ratio)
		self.f_sse = SSE(l2_value=self.l2_value)
		self.f_add = Add()

	def call(self, X):

		Y = self.f_add([self.f_cse(X), self.f_sse(X)])

		return Y

	def get_config(self):

		config = super().get_config()
		config["activation"] = self.activation
		config["num_out_filters"] = self.num_out_filters
		config["l2_value"] = self.l2_value
		config["ratio"] = self.ratio
		return config
