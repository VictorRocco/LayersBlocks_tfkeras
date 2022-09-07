# ==========================================
# === Residual Unet similar to RSU U2Net ===
# ==========================================

# sources:
# https://github.com/xuebinqin/U-2-Net
# https://arxiv.org/pdf/2005.09007.pdf

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2

from .StdCNA import StdCNA
from .CSE import CSE

@tf.keras.utils.register_keras_serializable()
class ResidualUnet(tf.keras.layers.Layer):

	def __init__(self, num_out_filters, num_unet_filters, num_layers,
				 function="StdCNA", #StdCNA (only one option at the moment)
				 aggregation="Concatenate", #Concatenate / Add
				 residual=True, #True=std residual, False=not residual, like U2NET
				 output_CSE=False,  # False / True
				 output_dropout=None,  # None or 0.xx
				 l2_value=0.001, **kwargs):

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
		for num_layer in range(self.num_layers):
			if num_layer == 0: #la 1ra (de mas arriba) out_filters, el resto unet_filters
				filters = num_out_filters
			else:
				filters = num_unet_filters
			self.f_encoder_stdcna[num_layer] = StdCNA(num_out_filters=filters, l2_value=self.l2_value)
			self.f_maxpool[num_layer] = MaxPool2D((2, 2)) # Downsampling

		#Bottleneck
		self.f_bottleneck = StdCNA(num_out_filters=num_unet_filters, l2_value=self.l2_value)
		self.f_dilated_bottleneck = StdCNA(num_out_filters=num_unet_filters, dilation_rate=(2, 2), l2_value=self.l2_value)
		if self.aggregation == "Concatenate":
			self.f_bottleneck_aggregation = Concatenate()
		elif self.aggregation == "Add":
			self.f_bottleneck_aggregation = Add()
		else:
			self.f_bottleneck_aggregation = Concatenate()

		#Decoder
		self.f_upsampling = {}
		self.f_decoder_aggregation = {}
		self.f_decoder_stdcna = {}

		#Ajusto filtros el Deconv Add de la capa de mas arriba (sin activacion)
		if self.aggregation == "Add":
			self.f_decoder_conv2d_num_filters = Conv2D(filters=self.num_out_filters, kernel_size=(1, 1))

		for num_layer in reversed(range(self.num_layers)):
			self.f_upsampling[num_layer] = UpSampling2D((2, 2), interpolation='bilinear')
			if self.aggregation == "Concatenate":
				self.f_decoder_aggregation[num_layer] = Concatenate()
			elif self.aggregation == "Add":
				self.f_decoder_aggregation[num_layer] = Add()
			else:
				self.f_decoder_aggregation[num_layer] = Concatenate()
			if num_layer == 0: #la 1ra (de mas arriba) out_filters, el resto unet_filters
				filters = num_out_filters
			else:
				filters = num_unet_filters
			self.f_decoder_stdcna[num_layer] = StdCNA(num_out_filters=filters, l2_value=self.l2_value)

		#Output
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
			self.f_output_conv2d_num_filters = Conv2D(filters=self.num_out_filters, kernel_size=(1, 1))

	def call(self, X):

		Y = X

		#(Residual) Input
		Y = self.f_stdcna_in(Y)

		#Encoder
		for num_layer in range(self.num_layers):
			Y = self.f_encoder_stdcna[num_layer](Y)
			self.skip_connection[num_layer] = Y
			Y = self.f_maxpool[num_layer](Y) # Downsampling

		# Bridge
		self.bottleneck_result = self.f_bottleneck(Y)
		self.dilated_bottleneck_result = self.f_dilated_bottleneck(Y)
		Y = self.f_bottleneck_aggregation([self.bottleneck_result, self.dilated_bottleneck_result])

		# Decoder
		for num_layer in reversed(range(self.num_layers)):
			Y = self.f_upsampling[num_layer](Y)
			# Ajusto filtros el Deconv Add de la capa de mas arriba
			if (self.aggregation == "Add") and (Y.shape[-1] != self.skip_connection[num_layer].shape[-1]):
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
