import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.layers import LeakyReLU
import xarray as xr
import random
from pathlib import Path
import time
from datetime import datetime
from packaging import version
from keras import initializers
import os
from tensorflow_addons.metrics import RSquare


#All the hyperparameters here are coming from the ClimSim hyperparameters optimization

in_shape  = (60, 6)
out_shape = (60, 10)

output_length_lin  = 2
output_length_relu = out_shape[-1] - 2

depth              = 12
channel_width      = 406
kernel_width       = 3
activation         = "relu"
pre_out_activation = "elu"
dropout            = 0.175
optimizer          = "Adam"
loss               = "mean_absolute_error"

channel_dims = [hp_channel_width] * hp_depth
kernels      = [hp_kernel_width] * hp_depth


def mae_adjusted(y_true, y_pred):
    ae = K.abs(y_pred - y_true)
    return K.mean(ae[:,:,0:2])*(120/128) + K.mean(ae[:,:,2:10])*(8/128)



def build_and_compile_model(n_samples):

    # Initialize model architecture
    input_layer = keras.Input(shape=in_shape)
    x = input_layer  # Set aside input layer
    previous_block_activation = x  # Set aside residual
    for filters, kernel_size in zip(channel_dims, kernels):
        # First conv layer in block
        # 'same' applies zero padding.
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(x)
        # todo: add se_block
        x = keras.layers.Activation(activation)(x)
        x = keras.layers.Dropout(dropout)(x)

        # Second convolution layer
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(x)
        x = keras.layers.Activation(hp_activation)(x)
        x = keras.layers.Dropout(hp_dropout)(x)

        # Project residual
        residual = Conv1D(
            filters=filters, kernel_size=1, strides=1, padding="same"
        )(previous_block_activation)
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Output layers.
    # x = keras.layers.Dense(filters[-1], activation='gelu')(x) # Add another last layer.
    x = Conv1D(
        out_shape[-1],
        kernel_size=1,
        activation=pre_out_activation,
        padding="same",
    )(x)
    # Assume that vertically resolved variables follow no particular range.
    output_lin = keras.layers.Dense(output_length_lin, activation="linear")(x)
    # Assume that all globally resolved variables are positive.
    output_relu = keras.layers.Dense(output_length_relu, activation="relu")(x)
    output_layer = keras.layers.Concatenate()([output_lin, output_relu])

    model = keras.Model(input_layer, output_layer, name="cnn")

    # Optimizer
    # Set up cyclic learning rate
    INIT_LR = 1e-4
    MAX_LR = 1e-3
    steps_per_epoch = n_samples // hp_depth
    clr = tfa.optimizers.CyclicalLearningRate(
        initial_learning_rate=INIT_LR,
        maximal_learning_rate=MAX_LR,
        scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
        step_size=2 * steps_per_epoch,
        scale_mode="cycle",
    )

    # Set up optimizer
    my_optimizer = keras.optimizers.Adam(learning_rate=clr)
    loss = mae_adjusted

    model.compile(
                optimizer=my_optimizer,
                loss=loss,
                metrics=["mse", "mae", "accuracy", mse_adjusted, mae_adjusted, continuous_ranked_probability_score],
            )
    return model