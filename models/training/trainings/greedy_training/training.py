import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.layers import LeakyReLU
import sys
import xarray as xr
import glob
import random
from pathlib import Path
import time
from datetime import datetime
from packaging import version
from MLP.MLPv1 import build_and_compile_model

# training loading directly all the data #

#to make the training reproductible
np.random.seed(212)
random.seed(370)
tf.random.set_seed(925)


input_length = 124
output_length = 128
output_length_lin = 120
output_length_relu = 8
num_epochs = 4




batch_size= 3072

ClimSim_data = False

if (ClimSim_data):
    data_path = '/gpfsscratch/rech/psl/upu87pm/preprocessed_data/'
    input_training_data = np.load(data_path+'train_input.npy')
    target_training_data = np.load(data_path+'train_target.npy')

    input_val_data = np.load(data_path+'val_input.npy')
    target_val_data = np.load(data_path+'val_target.npy')

else:
    data_path = '/gpfsscratch/rech/psl/upu87pm/my_preprocessed_data/'


    ClimSim_train_x = glob.glob(data_path + 'training/60/input_*.npy')
    ClimSim_train_y = glob.glob(data_path + 'training/60/target_*.npy')

    ClimSim_val_x     = glob.glob(data_path + 'val/60/input_*.npy')
    ClimSim_val_y     = glob.glob(data_path + 'val/60/target_*.npy')

    print("training input data = ", ClimSim_train_x)


    input_training_data = np.load(ClimSim_train_x[0])
    target_training_data = np.load(ClimSim_train_y[0])

    input_val_data = np.load(ClimSim_val_x[0])
    target_val_data = np.load(ClimSim_val_y[0])


    for i in range(1, 60, 1):
        input_training_data = np.concatenate([input_training_data, np.load(ClimSim_train_x[i])])
        target_training_data = np.concatenate([target_training_data, np.load(ClimSim_train_y[i])])

        input_val_data = np.concatenate([input_val_data, np.load(ClimSim_val_x[i])])
        target_val_data = np.concatenate([target_val_data, np.load(ClimSim_val_y[i])])
    
    


n_samples = input_training_data.shape[0]



tds = tf.data.Dataset.from_tensor_slices((input_training_data, target_training_data))
tds_val = tf.data.Dataset.from_tensor_slices((input_val_data, target_val_data))

tds_shuffle_buffer = 384*30 # 30 day equivalent num_samples

tds = tds.shuffle(buffer_size=tds_shuffle_buffer, reshuffle_each_iteration=True)
tds = tds.batch(batch_size)
tds = tds.prefetch(buffer_size=int(np.ceil(tds_shuffle_buffer/batch_size))) # in realtion to the batch size

tds_val = tds_val.shuffle(buffer_size=tds_shuffle_buffer, reshuffle_each_iteration=True)
tds_val = tds_val.batch(batch_size)
tds_val = tds_val.prefetch(buffer_size=int(np.ceil(tds_shuffle_buffer/batch_size))) # in realtion to the batch size


model = build_and_compile_model(n_samples, batch_size)


checkpoint_dir = '/gpfswork/rech/psl/upu87pm/hybrid_climate_modelling/models/MLP/saved_models/greedy/'
    
checkpoint_best = keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "model2_epoch_{epoch}.h5",
                                                save_weights_only=False,
                                                verbose=1,
                                                monitor='val_loss',
                                                save_best_only=False) # first checkpoint for best model

#model1: Cmisim preprocessed data
#model2: my preprocessed data

earlystop = keras.callbacks.EarlyStopping('val_loss', patience=8)


model.fit(tds,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=tds_val,                   
        verbose=1,
        callbacks=[checkpoint_best, earlystop]
)

