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
from datetime import datetime
from packaging import version
from archi.MLPv1 import build_and_compile_model

# training based on what ClimSim did but with npyfiles#

#to make the training reproductible
# np.random.seed(212)
# random.seed(370)
# tf.random.set_seed(925)


num_epochs = 11
#n_samples = 26280
n_samples = 22526


#False to take ClimSim preprocessed data
my_data = True

v2_data = False

def train():

    # ClimSim generator 
    def load_py_dir_with_generator(filelist:list):
            def gen():
                for file in filelist:
                    # read inputs
                    ds = np.load(file)
                    # ds = np.delete(ds, 376, 1)
                    # print(ds.shape)

                    # read outputs
                    dso = np.load(file.replace('input','target'))


                    yield (ds, dso) # generating a tuple of (input, output)

            return tf.data.Dataset.from_generator(gen,
                                                output_types=(tf.float64, tf.float64),
                                                output_shapes=((None,input_length),(None,output_length))
    
                                            )


    print("loading data..")

    (model, model_name, input_length, output_length, batch_size) = build_and_compile_model(n_samples)


    if my_data:
        if v2_data:
            data_path = '/gpfsscratch/rech/psl/upu87pm/my_preprocessed_data/v2/'
            f_mli = glob.glob(data_path + 'training/72/shuffle_reduced_input_*.npy')
            f_mli_val = glob.glob(data_path + 'val/72/shuffle_reduced_input_*.npy')
            print("found ", len(f_mli), " training files and ", len(f_mli_val), "validation files")
        else:
            # data_path = '/gpfsscratch/rech/psl/upu87pm/my_preprocessed_data/'
            # f_mli = glob.glob(data_path + 'training/60/shuffle_input_*.npy')
            # f_mli_val = glob.glob(data_path + 'val/60/shuffle_input_*.npy')
            # print("found ", len(f_mli), " training files and ", len(f_mli_val), "validation files")
            data_path = '/gpfsscratch/rech/psl/upu87pm/my_preprocessed_data/v1/'
            f_mli = glob.glob(data_path + 'training/72/shuffle_reduced_input_*.npy')
            f_mli_val = glob.glob(data_path + 'val/72/shuffle_reduced_input_*.npy')
            print("found ", len(f_mli), " training files and ", len(f_mli_val), "validation files")

        


    else:
        print("has to be change !!")
        data_path = '/gpfsscratch/rech/psl/upu87pm/preprocessed_data/'
        f_mli = glob.glob(data_path + 'train_input.npy')
        f_mli_val = glob.glob(data_path + 'val_input.npy')



    
    tds_shuffle_buffer = 384*30 # 30 day equivalent num_samples
    tds = load_py_dir_with_generator(f_mli)
    tds = tds.unbatch()
    tds = tds.shuffle(buffer_size=tds_shuffle_buffer, reshuffle_each_iteration=True)
    tds = tds.batch(batch_size)
    tds = tds.prefetch(buffer_size=int(np.ceil(tds_shuffle_buffer/batch_size))) # in realtion to the batch size
    print("training set loaded")

    tds_val = load_py_dir_with_generator(f_mli_val)
    tds_val = tds_val.unbatch()
    tds_val = tds_val.shuffle(buffer_size=tds_shuffle_buffer, reshuffle_each_iteration=True)
    tds_val = tds_val.batch(batch_size)
    tds_val = tds_val.prefetch(buffer_size=int(np.ceil(tds_shuffle_buffer/batch_size))) # in realtion to the batch size
    print("validation set loaded")


    steps_per_epoch = n_samples // batch_size

    checkpoint_dir = '/gpfswork/rech/psl/upu87pm/hybrid_climate_modelling/models/saved_models/'+model_name+'/'


    checkpoint_best = keras.callbacks.ModelCheckpoint(filepath =checkpoint_dir + "model4_epoch_{epoch}.h5",
                                                      save_weights_only=False,
                                                      verbose=1,
                                                      monitor='val_loss',
                                                      save_best_only=False) # first checkpoint for best model
    
    #model1: ClimSim preprocessed dat
    #model2: My preprocessed data
    #model3: my preprocessed data shuffled bf stride
    #model4: my split of the data
    

    earlystop = keras.callbacks.EarlyStopping('val_loss', patience=8)

    model.fit(tds,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    validation_data=tds_val,                   
                    verbose=1,
                    callbacks=[checkpoint_best, earlystop]
    )


    return


def main():

    print("n_samples = ", n_samples*384)
    train()
    

if __name__ == '__main__':
    main()
    