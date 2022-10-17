# =============================================
# === Residual Unet similar to RSU in U2Net ===
# =============================================

# num_layers parameter
# ===================
# Set the number of layers (included both, the input layer and the bottleneck layer)
# Example: 6 layers = 5 layers (included the input layer) + bottleneck
# Be aware, parameter num_layers sets the minimum image shape in bottleneck
# You need to know the minimum image shape you want and then calculate num_layers
# Also in bottleneck, an additional dilated (2,2) lbConv2D is applied
# It is NOT recommended to use a minimum shape less than 8x8
# Formulae: minimum_shape = input_shape / 2^(num_layers-1)
# Example: input shape 256x256, num_layers = 6 -> minimum_shape = 256 / 2^(6-1) = 256 / 32 = 8
# Layer 1 (encoder)(same as input) = 256x256
# Layer 2 (encoder) = 128x128
# Layer 3 (encoder) = 64x64
# Layer 4 (encoder) = 32x32
# Layer 5 (encoder) = 16x16
# Layer 6 (bottleneck) = 8x8
# Additional (bottelenck): dilated (2,2) lbConv2D on 8x8

# sources:
# https://github.com/xuebinqin/U-2-Net
# https://arxiv.org/pdf/2005.09007.pdf

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2

from .StdCNA import StdCNA
from .CSE import CSE
from .lbConv2D import lbConv2D

@tf.keras.utils.register_keras_serializable()
class ResidualUnet(tf.keras.layers.Layer):

	def __init__(self, num_out_filters, num_unet_filters,
				 num_layers, #read upper documentation (in order to calculate minimum bottleneck shape)
				 function="StdCNA", #StdCNA (only one option at the moment)
				 aggregation="concatenate", #concatenate / add
				 residual=True, #True=std residual, False=not residual (like U2NET)
				 output_CSE=False, # False / True
				 output_dropout=None, # None or 0.xx
				 l2_value=None,
				 **kwargs):

		if function not in ("StdCNA"):
			raise ValueError("invalid argument: function = ", function)
		if aggregation not in ("concatenate", "add"):
			raise ValueError("invalid argument: aggregation = ", aggregation)
		if residual not in (True, False):
			raise ValueError("invalid argument: residual = ", residual)
		if output_CSE not in (True, False):
			raise ValueError("invalid argument: output_CSE = ", output_CSE)

		super().__init__(**kwargs)
		self.num_out_filters = num_out_filters
		self.num_unet_filters = num_unet_filters
		self.num_layers = num_layers
		self.function = function
		self.aggregation = aggregation
		self.residual = residual
		self.output_CSE = output_CSE
		self.output_dropout = output_dropout
		self.l2_value = l2_value

		#Residual Input
		self.f_stdcna_in = StdCNA(num_out_filters=self.num_out_filters, l2_value=self.l2_value)

		#Encoder
		self.f_encoder_stdcna = {}
		self.skip_connection = {}
		self.f_maxpool = {}
		for num_layer in range(self.num_layers - 1): #-1 = bottleneck
			if num_layer == 0: #la 1ra (de mas arriba) out_filters, el resto unet_filters
				filters = num_out_filters
			else:
				filters = num_unet_filters
			#print("ResidualUnet Encoder (num_layer, filters):", num_layer, filters, flush=True)
			self.f_encoder_stdcna[num_layer] = StdCNA(num_out_filters=filters, l2_value=self.l2_value)
			self.f_maxpool[num_layer] = MaxPool2D((2, 2)) # Downsampling

		#Bottleneck
		self.f_bottleneck = StdCNA(num_out_filters=num_unet_filters, l2_value=self.l2_value)
		#print("ResidualUnet Bottleneck", flush=True)
		self.f_dilated_bottleneck = StdCNA(num_out_filters=num_unet_filters, dilation_rate=(2, 2), l2_value=self.l2_value)
		#print("ResidualUnet Bottleneck dilated", flush=True)
		if self.aggregation == "concatenate":
			self.f_bottleneck_aggregation = Concatenate()
		elif self.aggregation == "add":
			self.f_bottleneck_aggregation = Add()
		else:
			assert "aggregation error"

		#Decoder
		self.f_upsampling = {}
		self.f_decoder_aggregation = {}
		self.f_decoder_stdcna = {}

		#Ajusto filtros el Deconv Add de la capa de mas arriba (sin activacion)
		if self.aggregation == "add":
			self.f_decoder_conv2d_num_filters = lbConv2D(num_out_filters=self.num_out_filters, kernel_size=(1, 1),
														 activation=None, l2_value=self.l2_value)

		for num_layer in reversed(range(self.num_layers - 1)): #-1 = bottleneck
			self.f_upsampling[num_layer] = UpSampling2D((2, 2), interpolation='bilinear')
			if self.aggregation == "concatenate":
				self.f_decoder_aggregation[num_layer] = Concatenate()
			elif self.aggregation == "add":
				self.f_decoder_aggregation[num_layer] = Add()
			else:
				assert "aggregation error"
			if num_layer == 0: #la 1ra (de mas arriba) out_filters, el resto unet_filters
				filters = num_out_filters
			else:
				filters = num_unet_filters
			#print("ResidualUnet Decoder (num_layer, filters):", num_layer, filters, flush=True)
			self.f_decoder_stdcna[num_layer] = StdCNA(num_out_filters=filters, l2_value=self.l2_value)

		#Output
		if self.residual == True:
			self.f_output_add = Add() #Residual Add

		if self.output_CSE is not None:
			self.f_output_CSE = CSE()

		if self.output_dropout is not None:
			self.f_output_dropout = Dropout(self.output_dropout)

	def build(self, input_shape):

		# Si es necesario ajusto la cantidad de filtros finales
		# para poder hacer el Residual ADD (sin activacion)
		self.input_channels = input_shape[-1]
		if self.num_out_filters != self.input_channels:
			self.f_output_conv2d_num_filters = lbConv2D(num_out_filters=self.num_out_filters, kernel_size=(1, 1),
														activation=None, l2_value=self.l2_value)

	def call(self, X):

		Y = X

		#(Residual) Input
		Y = self.f_stdcna_in(Y)

		#Encoder
		for num_layer in range(self.num_layers -1): #-1 = bottleneck
			Y = self.f_encoder_stdcna[num_layer](Y)
			self.skip_connection[num_layer] = Y
			Y = self.f_maxpool[num_layer](Y) # Downsampling

		# Bridge
		self.bottleneck_result = self.f_bottleneck(Y)
		self.dilated_bottleneck_result = self.f_dilated_bottleneck(Y)
		Y = self.f_bottleneck_aggregation([self.bottleneck_result, self.dilated_bottleneck_result])

		# Decoder
		for num_layer in reversed(range(self.num_layers -1)): #-1 = bottleneck
			Y = self.f_upsampling[num_layer](Y)
			# Ajusto filtros el Deconv Add de la capa de mas arriba
			if (self.aggregation == "add") and (Y.shape[-1] != self.skip_connection[num_layer].shape[-1]):
				Y = self.f_decoder_aggregation[num_layer]([self.f_decoder_conv2d_num_filters(Y),
														   self.skip_connection[num_layer]])
			else:
				Y = self.f_decoder_aggregation[num_layer]([Y, self.skip_connection[num_layer]])
			Y = self.f_decoder_stdcna[num_layer](Y)

		#Output

		if self.residual == True:
			# Si es necesario ajusto la cantidad de filtros finales para poder hacer el Residual ADD
			if self.num_out_filters != self.input_channels:
				Y = self.f_output_add([self.f_output_conv2d_num_filters(X), Y])
			else:
				Y = self.f_output_add([X, Y])

		if self.output_CSE is not None:
			Y = self.f_output_CSE(Y)

		if self.output_dropout is not None:
			Y = self.f_output_dropout(Y)

		return Y

	def get_config(self):

		config = super().get_config()
		config["num_out_filters"] = self.num_out_filters
		config["num_unet_filters"] = self.num_unet_filters
		config["num_layers"] = self.num_layers
		config["function"] = self.function
		config["aggregation"] = self.aggregation
		config["residual"] = self.residual
		config["output_CSE"] = self.output_CSE
		config["output_dropout"] = self.output_dropout
		config["l2_value"] = self.l2_value
		return config
