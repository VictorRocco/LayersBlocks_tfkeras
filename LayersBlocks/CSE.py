#Channel-wise Squeeze and Excite block

# https://arxiv.org/abs/1709.01507
# https://github.com/RayXie29/SENet_Kerashttps://github.com/RayXie29/SENet_Keras
# https://blog.paperspace.com/channel-attention-squeeze-and-excitation-networks/
# https://github.com/titu1994/keras-squeeze-excite-network

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2

@tf.keras.utils.register_keras_serializable()
class CSE(tf.keras.layers.Layer):

	def __init__(self, X_input_shape,
				 activation="LR010", #LR010=LeakyReLU(0.10), RELU=ReLU, None
				 l2_value=0.001, ratio=16, **kwargs):
        			
		super().__init__(**kwargs)
		self.set_config(X_input_shape, activation, l2_value, ratio)

		if self.activation == "LR010":
			self.f_activation = LeakyReLU(0.10)
		elif self.activation == "RELU":
			self.f_activation = ReLU()
		else:
			self.f_activation = None

		# Configuration
		input_channels = self.X_input_shape[-1]
		reduced_channels = input_channels // self.ratio
		if reduced_channels <= 1:
			reduced_channels = 2

		# Squeeze operation
		self.f_gap = GlobalAveragePooling2D()
		self.f_dense_1 = Dense(reduced_channels, kernel_initializer="he_normal", use_bias=False,
							   activation=self.f_activation,
							   kernel_regularizer=l2(self.l2_value), bias_regularizer=l2(self.l2_value))

		# Excitation operation
		self.f_dense_2 = Dense(input_channels, kernel_initializer="he_normal", use_bias=False,
							   activation='sigmoid',
							   kernel_regularizer=l2(self.l2_value), bias_regularizer=l2(self.l2_value))

		self.f_multiply = Multiply()

	def call(self, X):

		# Squeeze operation
		Y = self.f_gap(X)
		Y = self.f_dense_1(Y)

		# Excitation operation
		Y = self.f_dense_2(Y)

		Y = self.f_multiply([X, Y])

		return Y

	def set_config(self, X_input_shape,
				   activation="LR010", #LR010=LeakyReLU(0.10), RELU=ReLU, None
				   l2_value=0.001, ratio=16, **kwargs):

		self.X_input_shape = X_input_shape
		self.activation = activation
		self.l2_value = l2_value
		self.ratio = ratio

	def get_config(self):

		config = super().get_config()
		config["X_input_shape"] = self.X_input_shape
		config["activation"] = self.activation
		config["l2_value"] = self.l2_value
		config["ratio"] = self.ratio
		return config
