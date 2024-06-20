import xarray as xr
import numpy as np
import glob
import os
import random
import time
from multiprocessing import Pool, cpu_count, Manager

# np.random.seed(99)
# random.seed(27)

vars_mli = ['state_t','state_q0001','state_ps','pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
vars_mlo = ['ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC',
                'cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']


stride = 7

norm_path = '/gpfsdswork/projects/rech/psl/upu87pm/ClimSim/preprocessing/normalizations/'

mli_mean = xr.open_dataset(norm_path + 'inputs/input_mean.nc',  engine='netcdf4')
mli_max = xr.open_dataset(norm_path + 'inputs/input_max.nc',  engine='netcdf4')
mli_min = xr.open_dataset(norm_path + 'inputs/input_min.nc',  engine='netcdf4')
mlo_scale = xr.open_dataset(norm_path + 'outputs/output_scale.nc',  engine='netcdf4')

train_path_list = glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/000[1234567]-*/*.mli*.nc")\
                + glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/0008-01/*.mli*.nc")


train_path_list= sorted(train_path_list)
random.shuffle(train_path_list)           
train_path_list = train_path_list[::stride]

print("n_samples = ", len(train_path_list), "(1 sample = 1 snapshot = 1 file here)")


val_path_list = glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/0008-0[23456789]/*.mli*.nc")\
              + glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/0008-1[012]/*.mli*.nc")\
              + glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/0009-01/*.mli*.nc")

val_path_list= sorted(val_path_list)
random.shuffle(val_path_list)           
val_path_list = val_path_list[::stride]


n_npy = 60





def task(i):

    N = len(list_file)
    file_per_npy = N//n_npy
    if i == 0: print("loading ", file_per_npy*2, "files  (input + output):  ", i+1, "/", n_npy)
    
    start_time = time.time()
    ds = [xr.open_dataset(list_file[j], engine='netcdf4') for j in range(i*file_per_npy, (i+1)*file_per_npy, 1)]
    ds = xr.concat(ds, dim = 'time_counter')
    ds  = ds[vars_mli]

    dso = [xr.open_dataset(list_file[j].replace(".mli.", ".mlo.")) for j in range(i*file_per_npy, (i+1)*file_per_npy, 1)]
    dso = xr.concat(dso, dim = 'time_counter')

    end_time = time.time()

    if i == 0: print(f'loading  took {end_time - start_time} s')
    if i == 0: print("make mlo variales: ptend_t and ptend_q0001")

    dso['ptend_t'] = (dso['state_t'] - ds['state_t'])/1200 # T tendency [K/s]
    dso['ptend_q0001'] = (dso['state_q0001'] - ds['state_q0001'])/1200 # Q tendency [kg/kg/s]
    dso = dso[vars_mlo]

    D_min = {'min_'+str(var):(ds[var].min()).to_numpy() for var in vars_mli}
    D_max = {'max_'+str(var):(ds[var].max()).to_numpy() for var in vars_mli}
    D_mean = {'mean_'+str(var):(ds[var].mean()).to_numpy() for var in vars_mli}

    if i == 0: print("normalizatoin, scaling")
    ds = (ds-mli_mean)/(mli_max-mli_min)
    dso = dso*mlo_scale 

    if i == 0: print("computing stats")


    # for var in vars_mli: D_min['min_'+str(var)+'_normalized'] = (ds[var].min()).to_numpy()
    # for var in vars_mli: D_max['min_'+str(var)+'_normalized'] = (ds[var].max()).to_numpy()
    # for var in vars_mli: D_mean['min_'+str(var)+'_normalized'] = (ds[var].mean()).to_numpy()



    if i == 0: print("stack and save as npy")

    ds = ds.stack({'batch':{'time_counter', 'ncol'}})
    ds = ds.to_stacked_array("mlvar", sample_dims=["batch"], name='mli')
    dso = dso.stack({'batch':{'time_counter', 'ncol'}})
    dso = dso.to_stacked_array("mlvar", sample_dims=["batch"], name='mlo')


    ds = np.array(ds)
    dso = np.array(dso)

    if i == 0: print("input shape  ", ds.shape)
    if i == 0: print("output shpae ", dso.shape)	

    np.save('/gpfsscratch/rech/psl/upu87pm/my_preprocessed_data/'+set_+'/'+str(n_npy)+'/shuffle_input_'+str(i)+'.npy', ds)
    np.save('/gpfsscratch/rech/psl/upu87pm/my_preprocessed_data/'+set_+'/'+str(n_npy)+'/shuffle_target_'+str(i)+'.npy', dso)

    return (D_min, D_max, D_mean)


def prepro():
    N = len(list_file)
    file_per_npy = N//n_npy
    print("found ", N, " nc files and put ", file_per_npy, " of them in each npy file") 
    with Pool() as pool:
        result = pool.map(task, range(n_npy))

        D_min = [result[i][0] for i in range(n_npy)]
        D_max = [result[i][1] for i in range(n_npy)]
        D_mean = [result[i][2] for i in range(n_npy)]

        min_ = {k: min(t.get(k, 0) for t in D_min) for k in set.union(*[set(t) for t in D_min])}
        max_ = {k: max(t.get(k, 0) for t in D_max) for k in set.union(*[set(t) for t in D_max])}
        mean_ = {k: sum(t.get(k, 0)/n_npy for t in D_mean) for k in set.union(*[set(t) for t in D_mean])}


    # for D in (min_, max_, mean_):
    #     for key in D:
    #         print(key, " : ", D[key])
           


print("preprocessing data..")
print("running on ", cpu_count(), " cpus")

n_npy_per_porc = n_npy//cpu_count()

set_ = 'training'
list_file = train_path_list
print("preprocessing training data")
prepro()


set_ = 'val'
list_file = val_path_list
print("preprocessing val data")
prepro()