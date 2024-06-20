import xarray as xr
import numpy as np
import glob
import os
import random
import time
from multiprocessing import Pool, cpu_count, Manager

# np.random.seed(99)
# random.seed(27)

my_normalization = False

#557 input scalars
vars_mli      = ['state_t','state_q0001', 'state_q0002', 'state_q0003', 'state_u', 'state_v',
                     'state_ps', 'pbuf_SOLIN','pbuf_LHFLX', 'pbuf_SHFLX',  'pbuf_TAUX', 'pbuf_TAUY', 'pbuf_COSZRS',
                     'cam_in_ALDIF', 'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR', 'cam_in_LWUP',
                     'cam_in_ICEFRAC', 'cam_in_LANDFRAC', 'cam_in_OCNFRAC', 'cam_in_SNOWHLAND']
                     


vars_mli_utls = ['pbuf_ozone', 'pbuf_CH4', 'pbuf_N2O']

#368 output scalars
vars_mlo      = ['ptend_t','ptend_q0001','ptend_q0002','ptend_q0003', 'ptend_u', 'ptend_v',
                     'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC',
                     'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']


stride = 7

if my_normalization:
    index = list(range(376)) + list(range(381, 397)) + list(range(440, 456)) + list(range(499,515))

    norm_path = '/gpfswork/rech/psl/upu87pm/hybrid_climate_modelling/preprocessing/normalization_factors/'
    mli_mean  = np.load(norm_path + 'input_mean.npy')[index]
    mlo_mean  = np.load(norm_path + 'output_mean.npy')
    mli_var   = np.load(norm_path + 'input_var.npy')[index]
    mlo_var   = np.load(norm_path + 'output_var.npy')

    print("nomralization factors shape: ", mli_mean.shape, mlo_var.shape)



else:
    norm_path = '/gpfsdswork/projects/rech/psl/upu87pm/ClimSim/preprocessing/normalizations/'

    mli_mean = xr.open_dataset(norm_path + 'inputs/input_mean.nc',  engine='netcdf4')
    mli_max = xr.open_dataset(norm_path + 'inputs/input_max.nc',  engine='netcdf4')
    mli_min = xr.open_dataset(norm_path + 'inputs/input_min.nc',  engine='netcdf4')
    mlo_scale = xr.open_dataset(norm_path + 'outputs/output_scale.nc',  engine='netcdf4')

    for k, kds in enumerate([mli_mean, mli_min, mli_max]):
            kds_utls = kds[vars_mli_utls].isel(lev=slice(5,21)).rename({'lev':'lev2'})
            kds = kds[vars_mli]
            kds = kds.merge(kds_utls)
            if k==0: mli_mean=kds
            if k==1: mli_min=kds
            if k==2: mli_max=kds



train_path_list = glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/000[123456]-*/*.mli*.nc")\
                + glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/0007-01/*.mli*.nc")


train_path_list= sorted(train_path_list)
train_path_list = train_path_list[::stride]
random.shuffle(train_path_list)           

print("train_samples = ", len(train_path_list), "(1 sample = 1 snapshot = 1 file here)")


val_path_list = glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/0007-0[23456789]/*.mli*.nc")\
              + glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/0007-1[012]/*.mli*.nc")\
              + glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/0008-01/*.mli*.nc")

val_path_list= sorted(val_path_list)
val_path_list = val_path_list[::stride]
random.shuffle(val_path_list)           

print("val_samples = ", len(val_path_list), "(1 sample = 1 snapshot = 1 file here)")

stride = 6
test_path_list = glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/0008-0[3456789]/*.mli*.nc")\
              + glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/0008-1[012]/*.mli*.nc")\
              + glob.glob(os.environ['DSDIR']+"/ClimSim_low-res/train/0009-01/*.mli*.nc")


test_path_list= sorted(test_path_list)
random.shuffle(val_path_list) 
test_path_list = test_path_list[::stride]
          

print("test_samples = ", len(test_path_list), "(1 sample = 1 snapshot = 1 file here)")





def task(i):

    N = len(list_file)
    file_per_npy = N//n_npy
    if (N%n_npy) != 0 and i == 0 : print("missing some file because ", n_npy, " is not factor of ", N)

    if i == 0: print("loading ", file_per_npy*2, "files  (input + output):  ", i+1, "/", n_npy)
    
    start_time = time.time()
    ds = [xr.open_dataset(list_file[j], engine='netcdf4') for j in range(i*file_per_npy, (i+1)*file_per_npy, 1)]

    ds = xr.concat(ds, dim = 'time_counter')
    ds_utls = ds[vars_mli_utls].isel(lev=slice(5,21)).rename({'lev':'lev2'})

    ds  = ds[vars_mli]

    ds = ds.merge(ds_utls)

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

    D_min = {'min_'+str(var):(ds[var].min()).to_numpy() for var in vars_mli}
    D_max = {'max_'+str(var):(ds[var].max()).to_numpy() for var in vars_mli}
    D_mean = {'mean_'+str(var):(ds[var].mean()).to_numpy() for var in vars_mli}

    if i == 0: print("normalizatoin, scaling")

    if my_normalization:
        ds  = (ds - mli_mean)/mli_var
        dso = (dso - mlo_mean)/mlo_var
    
    else:
        ds = (ds-mli_mean)/(mli_max-mli_min)
        dso = dso*mlo_scale 

    if i == 0: print("computing stats")


    for var in vars_mli: D_min['min_'+str(var)+'_normalized'] = (ds[var].min()).to_numpy()
    for var in vars_mli: D_max['min_'+str(var)+'_normalized'] = (ds[var].max()).to_numpy()
    for var in vars_mli: D_mean['min_'+str(var)+'_normalized'] = (ds[var].mean()).to_numpy()



    if i == 0: print("stack and save as npy")

    ds = ds.stack({'batch':{'time_counter', 'ncol'}})
    ds = ds.to_stacked_array("mlvar", sample_dims=["batch"], name='mli')
    dso = dso.stack({'batch':{'time_counter', 'ncol'}})
    dso = dso.to_stacked_array("mlvar", sample_dims=["batch"], name='mlo')


    ds = np.array(ds)
    dso = np.array(dso)

    if i == 0: print("input shape  ", ds.shape)
    if i == 0: print("output shpae ", dso.shape)	

    np.save('/gpfsscratch/rech/psl/upu87pm/my_preprocessed_data/v2/'+set_+'/'+str(n_npy)+'/shuffle_reduced_input_'+str(i).rjust(2,'0')+'.npy', ds)
    np.save('/gpfsscratch/rech/psl/upu87pm/my_preprocessed_data/v2/'+set_+'/'+str(n_npy)+'/shuffle_reduced_target_'+str(i).rjust(2,'0')+'.npy', dso)

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


    for D in (min_, max_, mean_):
        for key in D:
            print(key, " : ", D[key])
           


print("preprocessing data..")
print("running on ", cpu_count(), " cpus")
n_npy = 72
n_npy_per_porc = n_npy//cpu_count()

set_ = 'training'
list_file = train_path_list
print("preprocessing training data")
prepro()

n_npy = 72
set_ = 'val'
list_file = val_path_list
print("preprocessing val data")
prepro()


n_npy = 72
set_ = 'test'
list_file = test_path_list
print("preprocessing test data")
prepro()