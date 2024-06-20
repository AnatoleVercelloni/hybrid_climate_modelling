import random
import glob
import os
import xarray as xr
import numpy as np


norm_path = '/gpfsdswork/projects/rech/psl/upu87pm/ClimSim/preprocessing/normalizations/'

mli_mean = xr.open_dataset(norm_path + 'inputs/input_mean.nc',  engine='netcdf4')
mli_max = xr.open_dataset(norm_path + 'inputs/input_max.nc',  engine='netcdf4')
mli_min = xr.open_dataset(norm_path + 'inputs/input_min.nc',  engine='netcdf4')
mlo_scale = xr.open_dataset(norm_path + 'outputs/output_scale.nc',  engine='netcdf4')

n_npy = 60
set_ = 'training'

train_path_list = glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/000[1234567]-*/*.mli*.nc", recursive=True)\
                + glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/0008-01/*.mli*.nc", recursive=True)

stride = 7
train_path_list= sorted(train_path_list)
train_path_list = train_path_list[::stride]

n_samples = len(train_path_list)
print("n_samples = ", len(train_path_list), "(1 sample = 1 snapshot = 1 file here)")

var = 'state_t'

sample_idx_ = random.randint(0,365*72-1)
loc_ = random.randint(0,384-1)
lev_ = random.randint(0,60-1)

sample_idx = sample_idx_
loc = loc_
lev = lev_

print("checking the state_t at lev", lev_, " for sample n°", sample_idx_, " loc = ", loc_)

####### without preprocessing #######

nc_sample = train_path_list[sample_idx]
nc_sample = xr.open_dataset(nc_sample)
print("the value of the original file is ", np.array(nc_sample['state_t'][lev][loc]))

####### my preprocessing #######

n_file = sample_idx//(n_samples//(n_npy))
py_path_sample = '/gpfsscratch/rech/psl/upu87pm/my_preprocessed_data/'+set_+'/'+str(n_npy)+'/input_'+str(n_file)+'.npy'
py_sample = np.load(py_path_sample)
print("loading my sample from file n°", n_file, " and idx (",(sample_idx*384 + loc) - (n_file)*168192, ",", lev, ")" )

# print("check my shape: ", py_sample.shape, "60 times")
print("the value I made is ", py_sample[(sample_idx*384 + loc) - (n_file)*168192,lev]*(mli_max['state_t'][lev].to_numpy() - mli_min['state_t'][lev].to_numpy()) + mli_mean['state_t'][lev].to_numpy())


####### ClimSim preprocessing #######

cs_path_sample = '/gpfsscratch/rech/psl/upu87pm/preprocessed_data/train_input.npy'
cs_sample = np.load(cs_path_sample)

# print("check ClimSim shape: ", cs_sample.shape)
print("loading my sample from  idx (",sample_idx*384 + loc, ",", lev, ")" )


print("the value preprocessed by ClimSim is ", cs_sample[sample_idx*384 + loc,lev]*(mli_max['state_t'][lev].to_numpy() - mli_min['state_t'][lev].to_numpy()) + mli_mean['state_t'][lev].to_numpy())

###### my 26k preprocessing ########

# path = glob.glob('/gpfsscratch/rech/yvd/upu87pm/my_preprocessed_data/'+set_+'/26k/inputs**.npy')
# # path = sorted(path)
# sample = np.load(path[sample_idx])
# print("the value I made (26k) is ",  sample[loc][lev]*(mli_max['state_t'][lev].to_numpy() - mli_min['state_t'][lev].to_numpy()) + mli_mean['state_t'][lev].to_numpy())


