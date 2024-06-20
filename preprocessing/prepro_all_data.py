import xarray as xr
import numpy as np
import glob
import os
import random
import time
from multiprocessing import Pool, cpu_count, Manager
import sys

# np.random.seed(99)
# random.seed(27)

rank = 5

print("rank = ", rank)


#556 input scalars
vars_mli      = ['state_t','state_q0001', 'state_q0002', 'state_q0003', 'state_u', 'state_v',
                     'state_ps', 'pbuf_SOLIN','pbuf_LHFLX', 'pbuf_SHFLX',  'pbuf_TAUX', 'pbuf_TAUY', 'pbuf_COSZRS',
                     'cam_in_ALDIF', 'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR', 'cam_in_LWUP',
                     'cam_in_ICEFRAC', 'cam_in_LANDFRAC', 'cam_in_OCNFRAC', 'cam_in_SNOWHLAND',
                     'pbuf_ozone', 'pbuf_CH4', 'pbuf_N2O']

#368 output scalars
vars_mlo      = ['ptend_t','ptend_q0001','ptend_q0002','ptend_q0003', 'ptend_u', 'ptend_v',
                     'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC',
                     'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']



#subsampling in time
stride = 1

all_path_list = glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/*/*.mli*.nc")


train_path_list = glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/000[1234567]-*/*.mli*.nc")\
                + glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/0008-01/*.mli*.nc")


train_path_list= sorted(train_path_list)
random.shuffle(train_path_list)           
train_path_list = train_path_list[::stride]




val_path_list = glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/0008-0[23456789]/*.mli*.nc")\
              + glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/0008-1[012]/*.mli*.nc")\
              + glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/0009-01/*.mli*.nc")

val_path_list= sorted(val_path_list)
random.shuffle(val_path_list)           
val_path_list = val_path_list[::stride]


#the number of cpus -> number of files at the end
n_npy = 80




def task(i):

    N = len(list_file)

    if N%n_npy != 0:
        print("should take a number of proc that devide the amount of nc files !")
        

    file_per_npy = N//n_npy

    if i == 0: print("loading ", file_per_npy*2, "files  (input + output):  ", i+1, "/", n_npy)
    
    #we open the batch of files and merge them to an xarray dataset adding a time dimension
    start_time = time.time()
    ds = [xr.open_dataset(list_file[j], engine='netcdf4') for j in range(i*file_per_npy, (i+1)*file_per_npy, 1)]
    ds = xr.concat(ds, dim = 'time_counter')

    #We select the variables 
    ds  = ds[vars_mli]

    #same for output data
    dso = [xr.open_dataset(list_file[j].replace(".mli.", ".mlo.")) for j in range(i*file_per_npy, (i+1)*file_per_npy, 1)]
    dso = xr.concat(dso, dim = 'time_counter')

    end_time = time.time()

    if i == 0: print(f'loading  took {end_time - start_time} s')
    if i == 0: print("make mlo variales: ptend_t and ptend_q0001")

    #we have to construct the tendency variables for the output data
    dso['ptend_t']     = (dso['state_t']     - ds['state_t'])/1200     # T tendency [K/s]
    dso['ptend_q0001'] = (dso['state_q0001'] - ds['state_q0001'])/1200 # Q tendency [kg/kg/s]
    dso['ptend_q0002'] = (dso['state_q0002'] - ds['state_q0002'])/1200 # Q tendency [kg/kg/s]
    dso['ptend_q0003'] = (dso['state_q0003'] - ds['state_q0003'])/1200 # Q tendency [kg/kg/s]
    dso['ptend_u']     = (dso['state_u']     - ds['state_u'])/1200     # Q tendency [kg/kg/s]
    dso['ptend_v']     = (dso['state_v']     - ds['state_v'])/1200     # Q tendency [kg/kg/s]

    dso = dso[vars_mlo]

    #we are computing the mean (over time and ncol)to a future normalization: put everything in a dictionnary
    #---> one mean per lev variable 
    if i == 0: print("computing stats")

    D_inputs_mean = {str(var):(ds[var].mean(dim = ['time_counter', 'ncol'])).to_numpy() for var in vars_mli}
    D_outputs_mean = {str(var):(dso[var].mean(dim = ['time_counter', 'ncol'])).to_numpy() for var in vars_mlo}

    D_inputs_var = {str(var):(ds[var].var(dim = ['time_counter', 'ncol'])**2).to_numpy() for var in vars_mli}
    D_outputs_var = {str(var):(dso[var].var(dim = ['time_counter', 'ncol'])**2).to_numpy() for var in vars_mlo}

    if i == 0: print("little mean of state_t = ", D_inputs_mean['state_t'])

    #we stack the data to have something like (ncol x time) x (var)  where vertical variable are 60 different variables
    if i == 0: print("stack and save as npy")

    ds = ds.stack({'batch':{'time_counter', 'ncol'}})
    ds = ds.to_stacked_array("mlvar", sample_dims=["batch"], name='mli')
    dso = dso.stack({'batch':{'time_counter', 'ncol'}})
    dso = dso.to_stacked_array("mlvar", sample_dims=["batch"], name='mlo')

    #we save the data into a .npy file
    ds = np.array(ds)
    dso = np.array(dso)

    if i == 0: print("input shape  ", ds.shape)
    if i == 0: print("output shpae ", dso.shape)	

    save_path = '/gpfsscratch/rech/psl/upu87pm/my_preprocessed_data/'

    np.save(save_path + set_+'/'+str(n_npy*nodes)+'/input_'+str(rank*n_npy + i).rjust(3, '0') +'.npy', ds)
    np.save(save_path + set_+'/'+str(n_npy*nodes)+'/target_'+str(rank*n_npy + i).rjust(3, '0') +'.npy', dso)

    return (D_inputs_mean, D_outputs_mean, D_inputs_var, D_outputs_var)


def prepro():
    

    #we split the data processing between the n_npy procs available
    with Pool() as pool:
        result = pool.map(task, range(n_npy))

        #rearranging the means to have two lists of dictionary 
        D_inputs_mean = [result[i][0] for i in range(n_npy)]
        D_outputs_mean = [result[i][1] for i in range(n_npy)]
        D_inputs_var = [result[i][2] for i in range(n_npy)]
        D_outputs_var = [result[i][3] for i in range(n_npy)]

        #every mean is computing over the same amount of data so we can easily compute the global mean
        inputs_mean = {k: sum(t.get(k, 0)/n_npy for t in D_inputs_mean) for k in set.union(*[set(t) for t in D_inputs_mean])}
        outputs_mean = {k: sum(t.get(k, 0)/n_npy for t in D_outputs_mean) for k in set.union(*[set(t) for t in D_outputs_mean])}

        inputs_var = {k: sum(t.get(k, 0)/n_npy for t in D_inputs_var) for k in set.union(*[set(t) for t in D_inputs_var])}
        outputs_var = {k: sum(t.get(k, 0)/n_npy for t in D_outputs_var) for k in set.union(*[set(t) for t in D_outputs_var])}


    print("global mean of state_t = ", inputs_mean['state_t'])



    L_inputs_mean  = []
    L_outputs_mean = []
    L_inputs_var  = []
    L_outputs_var = []

    for k in vars_mlo:
        j = outputs_mean[k]
        j_ = outputs_var[k]

        if '__iter__' in dir(j):
            L_outputs_mean = L_outputs_mean + list(j)
            L_outputs_var = L_outputs_var + list(j_)
        else:
            L_outputs_mean.append(j)
            L_outputs_var.append(j_)

    for k in vars_mli:
        i = inputs_mean[k]
        i_ = inputs_var[k]
        if '__iter__' in dir(i):
            L_inputs_mean = L_inputs_mean + list(i)
            L_inputs_var  = L_inputs_var + list(i_ )

        else:
            L_inputs_mean.append(i)
            L_inputs_var.append(i_)

    print(np.array(L_inputs_mean).shape)
        
    np.save('/gpfsscratch/rech/psl/upu87pm/my_preprocessed_data/stats/inputs_mean_'+str(rank)+'.npy', np.array(L_inputs_mean))
    np.save('/gpfsscratch/rech/psl/upu87pm/my_preprocessed_data/stats/outputs_mean_'+str(rank)+'.npy', np.array(L_outputs_mean))
    np.save('/gpfsscratch/rech/psl/upu87pm/my_preprocessed_data/stats/inputs_var_'+str(rank)+'.npy', np.array(L_inputs_var))
    np.save('/gpfsscratch/rech/psl/upu87pm/my_preprocessed_data/stats/outputs_var_'+str(rank)+'.npy', np.array(L_outputs_var))
    print("mean and var saved")


# print("preprocessing data..")
# print("running on ", cpu_count(), " cpus")

# n_npy_per_porc = n_npy//cpu_count()

# set_ = 'training'
# print("n_samples = ", len(train_path_list), "(1 sample = 1 snapshot = 1 file here)")
# list_file = train_path_list
# print("preprocessing training data")
# prepro()

# set_ = 'val'
# list_file = val_path_list
# print("preprocessing val data")
# prepro()

set_ = 'all'
print("n_samples = ", len(all_path_list), "(1 sample = 1 snapshot = 1 file here)")
list_file = all_path_list
nodes = 6
N = len(list_file)//nodes
file_per_npy = N//n_npy
print ('argument list', sys.argv)

print("found ", N, " nc files from ",rank*N, " to ", (rank+1)*N, " and put ", file_per_npy, " of them in each npy file for node ", rank) 
list_file = list_file[rank*N:(rank+1)*N]



print("preprocessing all data")
prepro()