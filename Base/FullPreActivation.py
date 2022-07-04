# Full Pre Activation sequence (Normalization) + Activation + Conv2d
# https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow_addons.layers import InstanceNormalization


@tf.keras.utils.register_keras_serializable()
class FullPreActivation(tf.keras.layers.Layer):

	def __init__(self, num_out_filters, name_prefix,
					kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1),
					padding="same", activation=LeakyReLU(0.10),
        			normalization="IN", #IN=InstanceNormalization, BN=BatchNormalization, None
        			l2_value=0.001, **kwargs):
        			
		super().__init__(**kwargs)
		set_config(num_out_filters, name_prefix,
					kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1),
					padding="same", activation=LeakyReLU(0.10),
        			normalization="IN",
        			l2_value=0.001)

		if normalization == "IN":
    	    self.f_normalization = InstanceNormalization(axis=-1, center=True, scale=True,
        	        beta_initializer="random_uniform", gamma_initializer="random_uniform",
            	    name=name_prefix+"--FPA_INSTNORM")
		elif normalization == "BN":
	    	self.f_normalization = BatchNormalization(name=name_prefix+"--FPA_BN")
	    else:
	    	self.normalization == None

		self.f_activation = activation(name=name_prefix+"--FPA_ACTIV")

		self.f_conv2d = Conv2D(num_out_filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate,
			padding=padding, kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value),
			name=name_prefix+"--FPA_CONV2D")

	def call(self, X):
	
		if normalization != None:
			Y = self.f_normalization(X)
		else:
			Y = X
			
		Y = self.f_activation(Y)

		Y = self.f_conv2d(Y)
		
		return Y

	def set_config(self, num_out_filters, name_prefix,
					kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1),
					padding="same", activation=LeakyReLU(0.10),
        			normalization="IN", #IN=InstanceNormalization, BN=BatchNormalization, None
        			l2_value=0.001):
        			
		self.num_out_filters = num_out_filters
		self.name_prefix = name_prefix
		self.kernel_size = kernel_size
		self.strides = strides
		self.dilation_rate = dilation_rate
		self.padding = padding 
		self.activation = activation,
		self.normalization = normalization
		self.l2_value = l2_value

	def get_config(self):
	
		config = super().get_config()
		config["num_out_filters"] = self.num_out_filters
		config["name_prefix"] = self.name_prefix
		config["kernel_size"] = self.kernel_size
		config["strides"] = self.strides
		config["dilation_rate"] = self.dilation_rate
		config["padding"] = self.padding 
		config["activation"] = self.activation,
		config["normalization"] = self.normalization
		config["l2_value"] = self.l2_value
		return config

    
