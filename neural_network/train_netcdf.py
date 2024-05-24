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



input_length = 124
output_length = 128
output_length_lin = 120
output_length_relu = 8
batch_size = 3072
#n_samples = 4224 #to change !!
num_epochs = 4

reso = 'low'
data_path = '/gpfsdswork/dataset/ClimSim_low-res/train/'
norm_path = '../../ClimSim/preprocessing/normalizations/'
mli_mean = xr.open_dataset(norm_path + 'inputs/input_mean.nc')
mli_max = xr.open_dataset(norm_path + 'inputs/input_max.nc')
mli_min = xr.open_dataset(norm_path + 'inputs/input_min.nc')
mlo_scale = xr.open_dataset(norm_path + 'outputs/output_scale.nc')


vars_mli = ['state_t','state_q0001','state_ps','pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
vars_mlo = ['ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC',
                'cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']

fn_retrained_best = f'best.h5'

stride_sample = 7 # prime number to sample all 'tod'
f_mli1 = glob.glob(data_path + '000[1234567]-*/E3SM-MMF.mli.*-*-*-*.nc')
f_mli2 = glob.glob(data_path + '0008-01/E3SM-MMF.mli.0008-01-*-*.nc')
f_mli = sorted([*f_mli1, *f_mli2])
random.shuffle(f_mli) # to reduce IO bottleneck
f_mli = f_mli[::stride_sample]
n_samples = len(f_mli)

# validation dataset for HPO
f_mli1 = glob.glob(data_path + '0008-0[23456789]/E3SM-MMF.mli.0008-0[23456789]-*-*.nc')
f_mli2 = glob.glob(data_path + '0008-1[012]/E3SM-MMF.mli.0008-1[012]-*-*.nc')
f_mli3 = glob.glob(data_path + '0009-01/E3SM-MMF.mli.0009-01-*-*.nc')
f_mli_val = sorted([*f_mli1, *f_mli2, *f_mli3])
random.shuffle(f_mli_val)
f_mli_val = f_mli_val[::stride_sample]

def set_environment():
    
    ## Part 2: Limit memory preallocation
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[:], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    ## Part 3: Query available GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # If there are multiple GPUs, you can iterate over them
        for gpu in gpus:
            print("GPU:", gpu)



def build_and_compile_model(n_samples):

    x = keras.layers.Input(shape=(input_length), name='input')
    input_layer = x

    N = [768, 640, 512, 640, 640]
    for n_units in N:
        x = keras.layers.Dense(n_units, activation=LeakyReLU(alpha=.15))(x)

    x = keras.layers.Dense(output_length, activation=LeakyReLU(alpha=.15))(x)

    output_lin   = keras.layers.Dense(output_length_lin,activation='linear')(x)
    output_relu  = keras.layers.Dense(output_length_relu,activation='relu')(x)
    output_layer = keras.layers.Concatenate()([output_lin, output_relu])


    model = keras.Model(input_layer, output_layer, name='MLPv1_model')

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
                    metrics=['mse','mae','accuracy'], 
                    run_eagerly=False)

    # model summary
    print(model.summary())

    return model

# @tf.function            
def train():

    print("eager execution : ", tf.executing_eagerly())


    


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

    

    
    #with strategy.scope():

    model = build_and_compile_model(n_samples)

    checkpoint_best = keras.callbacks.ModelCheckpoint(filepath=fn_retrained_best,
                                                      save_weights_only=False,
                                                      verbose=1,
                                                      monitor='val_loss',
                                                      save_best_only=True) # first checkpoint for best model

    earlystop = keras.callbacks.EarlyStopping('val_loss', patience=8)

    history = model.fit(tds,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    validation_data=tds_val,                   
                    verbose=1,
                    callbacks=[checkpoint_best, earlystop]
    )


    return history





def main():

    # set_environment()
    # strategy = tf.distribute.MirroredStrategy()


    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    print("n_samples = ", n_samples*384)
    print("validation set size = ", len(f_mli_val*384))
    # with strategy.scope():
    train()
    

if __name__ == '__main__':
    main()
    