# Full Pre Activation sequence: (Normalization) + (Activation) + Conv2d
# https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow_addons.layers import InstanceNormalization

from .Activation import Activation
from .lbConv2D import lbConv2D
from .Normalization import Normalization


@tf.keras.utils.register_keras_serializable()
class FullPreActivation(tf.keras.layers.Layer):

	def __init__(self, num_out_filters,
				 kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1),
				 padding="symmetric", # same, valid, symmetric, reflect
				 activation="LR010", #LR010=LeakyReLU(0.10), RELU=ReLU, None
				 normalization="IN", #IN=InstanceNormalization, BN=BatchNormalization, None
				 l2_value=None, **kwargs):

		# assert padding: checked on lbConv2D
		# assert activation: checked on Activation
		# assert normalization: checked on Normalization

		super().__init__(**kwargs)
		self.num_out_filters = num_out_filters
		self.kernel_size = kernel_size
		self.strides = strides
		self.dilation_rate = dilation_rate
		self.padding = padding
		self.activation = activation,
		self.normalization = normalization
		self.l2_value = l2_value

		self.f_normalization = Normalization(normalization=self.normalization)
		self.f_activation = Activation(activation=self.activation)
		self.f_conv2d = lbConv2D(num_out_filters=self.num_out_filters, kernel_size=self.kernel_size,
								 strides=self.strides, dilation_rate=self.dilation_rate,
								 padding=self.padding, activation=None, l2_value=self.l2_value)

	def call(self, X):

		Y = X
		Y = self.f_normalization(Y)
		Y = self.f_activation(Y)
		Y = self.f_conv2d(Y)
		return Y

	def get_config(self):

		config = super().get_config()
		config["num_out_filters"] = self.num_out_filters
		config["kernel_size"] = self.kernel_size
		config["strides"] = self.strides
		config["dilation_rate"] = self.dilation_rate
		config["padding"] = self.padding
		config["activation"] = self.activation
		config["normalization"] = self.normalization
		config["l2_value"] = self.l2_value
		return config


