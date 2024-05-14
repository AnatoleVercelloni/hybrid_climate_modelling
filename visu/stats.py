import xarray as xr
import psyplot.project as psy
import psyplot
import numpy as np

def load_ncfiles(filelist, grid_b):

    grid = psyplot.open_dataset(grid_b)

    print(grid)

    ncol = len(grid['ncol'])
    lev = len(grid['lev'])
    T = len(filelist)
    print(T)
    nvert = len(grid['nvertex'])

    inputs = [xr.open_dataset(file) for file in filelist]
    In = xr.concat(inputs, dim = 'time_counter')
    # In.expand_dims(dim = {'time_counter' : T})
    In['time_counter'] = range(T)

    
    In = xr.merge([grid, In])

    
    for var in In.variables:
        if str(var) == 'area': continue
        if (In[str(var)].shape) == (ncol, ): In[str(var)].attrs = {"coordinates": "lon lat"}
        elif (In[str(var)].shape) == (ncol, nvert) : In[str(var)].attrs = {"coordinates": "lon lat"}
        elif (In[str(var)].shape) == (T, ncol) : In[str(var)].attrs = {"coordinates": "time_centered lon lat"}
        elif (In[str(var)].shape) == (T, lev, ncol) : In[str(var)].attrs = {"coordinates": "time_centered lev lon lat"}

        # print(var, In [str(var)].shape)


    In['lon'].attrs = {"bounds": "bounds_lon"}
    In['lat'].attrs = {"bounds": "bounds_lat"}



    outputs = [xr.open_dataset(file.replace('.mli.','.mlo.')) for file in filelist]
    Ou = xr.concat(outputs, dim = 'time_counter')
    Ou['time_counter'] = range(T)

    Ou = xr.merge([grid, Ou])

    Ou['ptend_t'] = (Ou['state_t'] - In['state_t'])/1200
    Ou['ptend_q0001'] = (Ou['state_q0001'] - In['state_q0001'])/1200


    # print(In['state_t'][0][59])

    for var in Ou.variables:
        if str(var) == 'area': continue
        if (Ou[str(var)].shape) == (ncol, ): Ou[str(var)].attrs = {"coordinates": "lon lat"}
        elif (Ou[str(var)].shape) == (ncol, nvert) : Ou[str(var)].attrs = {"coordinates": "lon lat"}
        elif (Ou[str(var)].shape) == (T, ncol) : Ou[str(var)].attrs = {"coordinates": "time_centered lon lat"}
        elif (Ou[str(var)].shape) == (T, lev, ncol) : Ou[str(var)].attrs = {"coordinates": "time_centered lev lon lat"}

    Ou['lon'].attrs = {"bounds": "bounds_lon"}
    Ou['lat'].attrs = {"bounds": "bounds_lat"}


    return In, Ou


def npy_toxarray(file, grid_b):

    data = np.load(file)

    T = data.shape[0]%384

    ptend_t = data[:,:60].reshape((1,60,384))
    ptend_q0001 = data[:,60:120].reshape((1,60,384))
    cam_out_NETSW
    cam_out_FLWDS
    cam_out_PRECSC
    cam_out_PRECC
    cam_out_SOLS
    cam_out_SOLL
    cam_out_SOLSD
    cam_out_SOLLD

    print(t.shape)


#     ds = xr.Dataset(

#     data_vars={

#         'ptend_t': ('time_counter' , 'lev' , 'ncol'), data[]
    
#     },

#     coords=dict(

#         lon=("loc", lon),

#         lat=("loc", lat),

#         instrument=instruments,

#         time=time,

#         reference_time=reference_time,

#     ),

#     attrs=dict(description="Weather related data."),

# )
    pred = xr.DataArray(data)


    # pred_ = pred.to_dataset(dim = 'dim_1')
    # pred_.chunk({'dim_1' : 64, 'dim_1': 64})
    # pred = pred_.rename_dims({'dim_0': 'ncol'})

    return pred

