# =========================
# === Up Sample Like 2D ===
# =========================
# Up sampling layer from (N, H1, W1, C1) to (N, H2, W2, C1)
# Input tensor up sampled to have the same spatial size as target.

import tensorflow as tf
from tensorflow.keras.layers import *

@tf.keras.utils.register_keras_serializable()
class UpSampleLike2D(tf.keras.layers.Layer):

	def __init__(self, target_spatial_size=(256, 256), interpolation='bilinear', **kwargs):
		self.target_spatial_size = target_spatial_size
		self.interpolation = interpolation
		super().__init__(**kwargs)

	def build(self, input_shape):
		assert input_shape.ndims == 4

		self._input_H = input_shape[1]
		self._input_W = input_shape[2]

	def call(self, X):
		_upsampling_factor_H = self.target_spatial_size[0] / self._input_H
		_upsampling_factor_W = self.target_spatial_size[1] / self._input_W

		Y = X
		Y = UpSampling2D((_upsampling_factor_H, _upsampling_factor_W),
						 interpolation=self.interpolation)(Y)
		return Y

	def get_config(self):
		config = super().get_config()
		config["target_spatial_size"] = self.target_spatial_size
		config["interpolation"] = self.interpolation
		return config
