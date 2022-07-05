# Full Pre Activation sequence: (Normalization) + (Activation) + Conv2d
# https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow_addons.layers import InstanceNormalization

@tf.keras.utils.register_keras_serializable()
class FullPreActivation(tf.keras.layers.Layer):

	def __init__(self, num_out_filters, name_prefix=None,
				 kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1),
				 padding="same", activation="LR010", #LR010=LeakyReLU(0.10), RELU=ReLU, None
				 normalization="IN", #IN=InstanceNormalization, BN=BatchNormalization, None
				 l2_value=0.001, **kwargs):
        			
		super().__init__(name=str(name_prefix)+"_FPA", **kwargs)
		self.num_out_filters = num_out_filters
		self.name_prefix = name_prefix
		self.kernel_size = kernel_size
		self.strides = strides
		self.dilation_rate = dilation_rate
		self.padding = padding
		self.activation = activation,
		self.normalization = normalization
		self.l2_value = l2_value

		if normalization == "IN":
			self.f_normalization = InstanceNormalization(axis=-1, center=True, scale=True,
														 beta_initializer="random_uniform",
														 gamma_initializer="random_uniform")
		elif normalization == "BN":
			self.f_normalization = BatchNormalization()
		else:
			self.f_normalization = None

		if activation == "LR010":
			self.f_activation = LeakyReLU(0.10)
		elif activation == "RELU":
			self.f_activation = ReLU()
		else:
			self.f_activation = None

		self.f_conv2d = Conv2D(filters=num_out_filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate,
							   padding=padding, kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))

	def call(self, X):
	
		if self.f_normalization != None:
			Y = self.f_normalization(X)
		else:
			Y = X

		if self.f_activation != None:
			Y = self.f_activation(Y)
		else:
			Y = X

		Y = self.f_conv2d(Y)
		
		return Y

	def get_config(self):

		config = super().get_config()
		config["num_out_filters"] = self.num_out_filters
		config["kernel_size"] = self.kernel_size
		config["strides"] = self.strides
		config["dilation_rate"] = self.dilation_rate
		config["padding"] = self.padding
		config["activation"] = self.activation,
		config["normalization"] = self.normalization
		config["l2_value"] = self.l2_value
		return config


