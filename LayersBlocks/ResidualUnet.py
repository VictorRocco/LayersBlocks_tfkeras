# Standard Conv2d + Normalization + Activation

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2

from .StdCNA import StdCNA
from .CSE import CSE

@tf.keras.utils.register_keras_serializable()
class ResidualUnet(tf.keras.layers.Layer):

	def __init__(self, num_out_filters, num_unet_filters, num_layers,
				 function="StdCNA", #StdCNA (only one option at the moment)
				 output_CSE=False,  # False / True
				 output_dropout=None,  # None or 0.xx
				 l2_value=0.001, **kwargs):

		super().__init__(**kwargs)
		self.num_out_filters = num_out_filters
		self.num_unet_filters = num_unet_filters
		self.num_layers = num_layers
		self.function = function
		self.output_CSE = output_CSE
		self.output_dropout = output_dropout
		self.l2_value = l2_value

		#Input
		self.f_stdcna_in = StdCNA(num_out_filters=self.num_out_filters, l2_value=self.l2_value)

		#Downscale path
		self.f_downscale_stdcna = {}
		self.skip_connection = {}
		self.f_maxpool = {}
		for num_layer in range(self.num_layers):
			self.f_downscale_stdcna[num_layer] = StdCNA(num_out_filters=num_unet_filters)
			self.f_maxpool[num_layer] = MaxPool2D((2, 2)) # Downsampling

		#Bridge
		self.f_bridge = StdCNA(num_out_filters=num_unet_filters)
		self.f_dilated_bridge = StdCNA(num_out_filters=num_unet_filters)
		self.f_bridge_add = Add()

		#Upscale path
		self.f_upsampling = {}
		self.f_upscale_add = {}
		self.f_upscale_stdcna = {}
		for num_layer in reversed(range(self.num_layers)):
			self.f_upsampling = UpSampling2D((2, 2), interpolation='bilinear')
			self.f_upscale_add = Add()
			if num_layer == 0: #la 1ra (de mas arriba) out_filters, el resto unet_filters
				filters = num_out_filters
			else:
				filters = num_unet_filters
			self.f_upscale_stdcna = StdCNA(num_out_filters=filters)

		#Output
		self.f_output_add = Add()
		if self.output_CSE is not None:
			self.f_otuput_CSE = CSE()

		if self.output_dropout is not None:
			self.f_output_dropout = Dropout(self.output_dropout)

	def call(self, X):

		self.Y = X

		#Input
		self.Y = self.f_stdcna_in(self.Y)

		#Downscale path
		for num_layer in range(self.num_layers):
			self.Y = self.f_downscale_stdcna[num_layer](self.Y)
			self.skip_connection[num_layer] = self.Y
			self.Y = self.f_maxpool[num_layer](self.Y) # Downsampling

		# Bridge
		self.bridge_result = self.f_bridge(self.Y)
		self.dilated_bridge_result = self.f_dilated_bridge(self.Y)
		self.Y = self.f_bridge_add([self.bridge_result, self.dilated_bridge_result])

		# Upscale path
		for num_layer in reversed(range(self.num_layers)):
			self.Y = self.f_upsampling(self.Y)
			self.Y = self.f_upscale_add([self.Y, self.skip_connection[num_layer]])
			if num_layer == 0: #la 1ra (de mas arriba) out_filters, el resto unet_filters
				filters = self.num_out_filters
			else:
				filters = self.num_unet_filters
			self.Y = self.f_upscale_stdcna(self.Y)

		#Output
		self.Y = self.f_output_add([self.Y, X]) #Residual Add

		if self.output_CSE is not None:
			self.Y = self.f_otuput_CSE(self.Y)

		if self.output_dropout is not None:
			self.Y = self.f_output_dropout(self.Y)

		return self.Y

	def get_config(self):

		config = super().get_config()
		config["num_out_filters"] = self.num_out_filters
		config["num_unet_filters"] = self.num_unet_filters
		config["num_layers"] = self.num_layers
		config["function"] = self.function
		config["output_CSE"] = self.output_CSE
		config["output_dropout"] = self.output_dropout
		config["l2_value"] = self.l2_value
		return config

"""
def internal_unet_block(input, num_out_filters, num_unet_filters, num_layers,
                        kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1),
                        padding="same", activation="LR010",  # LR010=LeakyReLU(0.10), RELU=ReLU, None
                        output_SE = False, #False / True
                        output_dropout=None, #None or 0.xx
                        l2_value = 0.001):

	skip_connection = []
	last_fnc = input

	last_fnc = StdCNA(num_out_filters=num_out_filters)(last_fnc)

	#print("out_filters/ubet_filters / layers:", num_out_filters, num_unet_filters, num_layers, flush=True)
	#print("1st Conv2D:", last_fnc.shape, flush=True)

	#Downscale path
	for num_layer in range(num_layers):
		#print("Down in:", num_layer, last_fnc.shape, flush=True)

		last_fnc = StdCNA(num_out_filters=num_unet_filters)(last_fnc)

		skip_connection.append(last_fnc)
		#print("skip_conn:", num_layer, last_fnc.shape, flush=True)

		last_fnc = MaxPool2D((2, 2))(last_fnc)  # Downsampling
		#print("Down out:", num_layer, last_fnc.shape, flush=True)

	#Bridge
	bridge = StdCNA(num_out_filters=num_unet_filters)(last_fnc)
	half_bridge = StdCNA(num_out_filters=num_unet_filters)(last_fnc)
	last_fnc = Add()([bridge, half_bridge])
	#print("bridge:", last_fnc.shape, flush=True)

	#Upscale path
	for num_layer in reversed(range(num_layers)):
		#print("Up in:", num_layer, last_fnc.shape, skip_connection[-1].shape, flush=True)

		#from Down and SkipConnection
		last_fnc = UpSampling2D((2, 2), interpolation='bilinear')(last_fnc)
		#print("Up upsample:", num_layer, last_fnc.shape, skip_connection[-1].shape, flush=True)
		last_fnc = Add()([last_fnc, skip_connection.pop()])

		if num_layer == 0: #la 1ra (de mas arriba) out_filters, el resto unet_filters
			filters = num_out_filters
		else:
			filters = num_unet_filters

		last_fnc = StdCNA(num_out_filters=filters)(last_fnc)
		#print("Up out:", num_layer, last_fnc.shape, flush=True)

	#residual Add
	last_fnc = Add()([input, last_fnc])

	if output_dropout is not None:
		last_fnc = Dropout(output_dropout)(last_fnc)

	#print("Return:", last_fnc.shape, flush=True)
	return last_fnc
"""