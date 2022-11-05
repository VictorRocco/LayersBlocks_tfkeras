# ==============================
# === SubPixel UpSampling 2D ===
# ==============================
# Up sampling layer from (N, H, W, C) to (N, H*factor, W*factor, C/(factor**2))
# If number of channels C < (factor**2), we compute SubPixelUpSampling with
# available channels and then UpSampling2D with the remaining scale.
# Example: scale = 16 but we have 64 channels, then max scale is 8, so we compute
# SubPixel upsampling with factor 8 and then UpSampling2D with scale 2,
# So we get the total scale of 16 (8*2=16).

# REFERENCE:
# Paper:
# 	Real-Time Single Image and Video Super-Resolution Using an Efficient
#   Sub-Pixel Convolutional Neural Network Shi et Al.
#   https://arxiv.org/abs/1609.05158
# Base code:
# 	https://github.com/fengwang/subpixel_conv2d/blob/master/subpixel_conv2d.py

import tensorflow as tf
from tensorflow.keras.layers import *


@tf.keras.utils.register_keras_serializable()
class SubPixelUpSampling2D(tf.keras.layers.Layer):
    def __init__(self, upsampling_factor=2, **kwargs):
        super().__init__(**kwargs)
        self.upsampling_factor = upsampling_factor

    def _subpixelupsampling_factor_calculation(self, channels: int) -> bool:
        factor = 2
        for index in [2, 4, 8, 16, 32, 64, 128]:
            if (index**2) <= channels:
                factor = index
            else:
                break
        return factor

    def build(self, input_shape):
        self._channels = input_shape[-1]
        self._squared_factor = self.upsampling_factor * self.upsampling_factor
        self._subpixelupsampling_factor = self._subpixelupsampling_factor_calculation(
            self._channels
        )
        self._remainder_factor = (
            self.upsampling_factor / self._subpixelupsampling_factor
        )

        # print("==============")
        # print("_channels:", self._channels)
        # print("upsampling_factor:", self.upsampling_factor)
        # print("_squared_factor:", self._squared_factor)
        # print("_subpixelupsampling_factor:", self._subpixelupsampling_factor)
        # print("_remainder_factor:", self._remainder_factor)

        if self.upsampling_factor < 1:
            raise ValueError(
                "upsampling_factor should be integer >= 1, received: "
                + str(self.upsampling_factor)
            )

        if (self.upsampling_factor > self._subpixelupsampling_factor) and (
            self._remainder_factor.is_integer() is False
        ):
            raise ValueError(
                "Number of channels "
                + str(self._channels)
                + " not suitable for SubPixelUpSampling. "
                + "Input shape is: "
                + str(input_shape)
            )

        if self._channels < self._squared_factor:
            if self.upsampling_factor % self._subpixelupsampling_factor != 0:
                raise ValueError(
                    "Number of channels "
                    + str(self._channels)
                    + " does not match upsampling_factor "
                    + str(self.upsampling_factor)
                    + " relation. "
                    + "Input shape is: "
                    + str(input_shape)
                )
        else:
            if (self._channels % self._squared_factor) != 0:
                raise ValueError(
                    "Number of channels "
                    + str(self._channels)
                    + " is not a (integer) multiple of upsampling_factor**2 "
                    + str(self._squared_factor)
                    + ". Input shape is: "
                    + str(input_shape)
                    + ". upsampling_factor is: "
                    + str(self.upsampling_factor)
                )

    def compute_output_shape(self, input_shape):
        self._channels

        if self.upsampling_factor == 1:
            output_shape = input_shape
            # print("output_shape 1:", output_shape)
            return output_shape
        elif self._remainder_factor > 1:
            output_shape = (
                input_shape[0],
                input_shape[1] * self.upsampling_factor
                if input_shape[1] is not None
                else None,
                input_shape[2] * self.upsampling_factor
                if input_shape[2] is not None
                else None,
                int(
                    input_shape[3]
                    / (
                        self._subpixelupsampling_factor
                        * self._subpixelupsampling_factor
                    )
                ),
            )
            # print("output_shape 2:", output_shape)
            return output_shape
        else:
            output_shape = (
                input_shape[0],
                input_shape[1] * self.upsampling_factor
                if input_shape[1] is not None
                else None,
                input_shape[2] * self.upsampling_factor
                if input_shape[2] is not None
                else None,
                int(input_shape[3] / (self.upsampling_factor * self.upsampling_factor)),
            )
            # print("output_shape 3:", output_shape)
            return output_shape

    def call(self, X):
        Y = X
        if self.upsampling_factor == 1:
            pass
        elif self._remainder_factor > 1:
            Y = tf.nn.depth_to_space(Y, self._subpixelupsampling_factor)
            Y = UpSampling2D(
                (self._remainder_factor, self._remainder_factor),
                interpolation="bilinear",
            )(Y)
        else:
            Y = tf.nn.depth_to_space(Y, self.upsampling_factor)
        return Y

    def get_config(self):
        config = super().get_config()
        config["upsampling_factor"] = self.upsampling_factor
        return config
