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

#All the hyperparameters here are coming from the ClimSim hyperparameters optimization

#to make the training reproductible
# np.random.seed(212)
# random.seed(370)
# tf.random.set_seed(925)

input_length = 124
output_length = 128
output_length_lin = 120
output_length_relu = 8
n_samples = 26280
batch_size = 3072
model_name = 'MLPv1_model'



def build_and_compile_model(n_samples):

    x = keras.layers.Input(shape=(input_length), name='input')
    input_layer = x

    N = [768, 640, 512, 640, 640]
    for n_units in N:
        x = keras.layers.Dense(n_units, activation=LeakyReLU(alpha=.15))(x)

    x = keras.layers.Dense(output_length, activation=LeakyReLU(alpha=.15))(x)

    output_lin   = keras.layers.Dense(output_length_lin, activation='linear')(x)
    output_relu  = keras.layers.Dense(output_length_relu, activation='relu')(x)
    output_layer = keras.layers.Concatenate()([output_lin, output_relu])


    model = keras.Model(input_layer, output_layer, name=model_name)

    INIT_LR = 2.5e-4
    MAX_LR  = 2.5e-3
    steps_per_epoch = n_samples // batch_size    

    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=INIT_LR,
                                                maximal_learning_rate=MAX_LR,
                                                scale_fn = lambda x: 1/(2.**(x-1)),
                                                step_size = 2 * steps_per_epoch,
                                                scale_mode = 'cycle'
                                                )

    my_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=clr)


    model.compile(optimizer=my_optimizer, 
                    loss='mse',
                    metrics=['mse','mae','accuracy'])

    return (model, model_name, input_length, output_length, batch_size)