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
from MLP.MLPv1 import build_and_compile_model

# training based on what ClimSim did #

#to make the training reproductible
# np.random.seed(212)
# random.seed(370)
# tf.random.set_seed(925)

input_length = 124
output_length = 128
output_length_lin = 120
output_length_relu = 8
batch_size = 3072
num_epochs = 8

reso = 'low'
data_path = '/gpfsdswork/dataset/ClimSim_low-res/train/'
norm_path = '/gpfswork/rech/psl/upu87pm/ClimSim/preprocessing/normalizations/'
mli_mean = xr.open_dataset(norm_path + 'inputs/input_mean.nc', engine='netcdf4')
mli_max = xr.open_dataset(norm_path + 'inputs/input_max.nc',  engine='netcdf4')
mli_min = xr.open_dataset(norm_path + 'inputs/input_min.nc',  engine='netcdf4')
mlo_scale = xr.open_dataset(norm_path + 'outputs/output_scale.nc',  engine='netcdf4')


vars_mli = ['state_t','state_q0001','state_ps','pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
vars_mlo = ['ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC',
                'cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']

checkpoint_dir = '/gpfswork/rech/psl/upu87pm/hybrid_climate_modelling/models/MLP/saved_models/ClimSim/'

stride_sample = 7 # prime number to sample all 'tod'
f_mli1 = glob.glob(data_path + '000[1234567]-*/E3SM-MMF.mli.*-*-*-*.nc')
f_mli2 = glob.glob(data_path + '0008-01/E3SM-MMF.mli.0008-01-*-*.nc')
f_mli = sorted([*f_mli1, *f_mli2])
random.shuffle(f_mli) # to reduce IO bottleneck
f_mli = f_mli[::stride_sample]
#random.shuffle(f_mli) # Here, we are shuffling after downsampling in time bc it makes more sense for me
n_samples = len(f_mli)  # we consider one sample = one snapshot of one col

# validation dataset for HPO
f_mli1 = glob.glob(data_path + '0008-0[23456789]/E3SM-MMF.mli.0008-0[23456789]-*-*.nc')
f_mli2 = glob.glob(data_path + '0008-1[012]/E3SM-MMF.mli.0008-1[012]-*-*.nc')
f_mli3 = glob.glob(data_path + '0009-01/E3SM-MMF.mli.0009-01-*-*.nc')
f_mli_val = sorted([*f_mli1, *f_mli2, *f_mli3])
random.shuffle(f_mli_val)
f_mli_val = f_mli_val[::stride_sample]
#random.shuffle(f_mli_val) # Here, we are shuffling after downsampling in time bc it makes more sense for me


     
def train():

    # ClimSim generator 
    def load_nc_dir_with_generator(filelist:list):
        def gen():
            for file in filelist:
                # read mli
                ds = xr.open_dataset(file, engine='netcdf4')
                ds = ds[vars_mli]

                # read mlo
                dso = xr.open_dataset(file.replace('.mli.','.mlo.'), engine='netcdf4')

                # make mlo variales: ptend_t and ptend_q0001
                dso['ptend_t'] = (dso['state_t'] - ds['state_t'])/1200 # T tendency [K/s]
                dso['ptend_q0001'] = (dso['state_q0001'] - ds['state_q0001'])/1200 # Q tendency [kg/kg/s]
                dso = dso[vars_mlo]

                # normalizatoin, scaling
                ds = (ds-mli_mean)/(mli_max-mli_min)
                dso = dso*mlo_scale

                # stack
                #ds = ds.stack({'batch':{'sample','ncol'}}) # this line was for data files that include 'sample' dimension
                ds = ds.stack({'batch':{'ncol'}})
                ds = ds.to_stacked_array("mlvar", sample_dims=["batch"], name='mli')
                #dso = dso.stack({'batch':{'sample','ncol'}})
                dso = dso.stack({'batch':{'ncol'}})
                dso = dso.to_stacked_array("mlvar", sample_dims=["batch"], name='mlo')

                yield (ds.values, dso.values) # generating a tuple of (input, output)

        return tf.data.Dataset.from_generator(gen,
                                              output_types=(tf.float64, tf.float64),
                                              output_shapes=((None,input_length),(None,output_length))
                                             )

    print("loading data..")
    
    tds_shuffle_buffer = 384*30 # 30 day equivalent num_samples
    tds = load_nc_dir_with_generator(f_mli)
    tds = tds.unbatch()
    tds = tds.shuffle(buffer_size=tds_shuffle_buffer, reshuffle_each_iteration=True)
    tds = tds.batch(batch_size)
    tds = tds.prefetch(buffer_size=int(np.ceil(tds_shuffle_buffer/batch_size))) # in realtion to the batch size
    print("training set loaded")

    tds_val = load_nc_dir_with_generator(f_mli_val)
    tds_val = tds_val.unbatch()
    tds_val = tds_val.shuffle(buffer_size=tds_shuffle_buffer, reshuffle_each_iteration=True)
    tds_val = tds_val.batch(batch_size)
    tds_val = tds_val.prefetch(buffer_size=int(np.ceil(tds_shuffle_buffer/batch_size))) # in realtion to the batch size
    print("validation set loaded")


    steps_per_epoch = n_samples // batch_size


    model = build_and_compile_model(n_samples, batch_size)

    checkpoint = keras.callbacks.ModelCheckpoint(filepath =checkpoint_dir + "model1_epoch_{epoch}.h5",
                                                      save_weights_only=False,
                                                      verbose=1,
                                                      monitor='val_loss',
                                                      save_best_only=False) # first checkpoint for best model
    
    #model1:

    earlystop = keras.callbacks.EarlyStopping('val_loss', patience=8)

    history = model.fit(tds,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        validation_data=tds_val,
                        verbose=2,
                        callbacks=[checkpoint, earlystop])


    return


def main():

    # print("n_samples = ", n_samples*384)
    # print("validation set size = ", len(f_mli_val*384))
    train()
    

if __name__ == '__main__':
    main()
    