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


    grid = psyplot.open_dataset(grid_b)

 
    ncol = len(grid['ncol'])
    T = data.shape[0]//ncol
    print(T)
    lev = len(grid['lev'])
    nvertex = len(grid['nvertex'])

    data = data[:T*ncol]

    area_ = np.array(grid['area'])
    lon_ = np.array(grid['lon'])
    lat_ = np.array(grid['lat'])
    bounds_lon_ = np.array(grid['bounds_lon'])
    bounds_lat_ = np.array(grid['bounds_lat'])

    ptend_t = data[:,:60].reshape((T,lev, ncol))
    ptend_q0001 = data[:,60:120].reshape((T, lev, ncol))
    cam_out_NETSW = data[:,120:121].reshape((T, ncol))
    cam_out_FLWDS = data[:,121:122].reshape((T, ncol))
    cam_out_PRECSC = data[:,122:123].reshape((T, ncol))
    cam_out_PRECC = data[:,123:124].reshape((T, ncol))
    cam_out_SOLS = data[:,124:125].reshape((T, ncol))
    cam_out_SOLL = data[:,125:126].reshape((T, ncol))
    cam_out_SOLSD = data[:,126:127].reshape((T, ncol))
    cam_out_SOLLD = data[:,127:128].reshape((T, ncol))

    # print(t.shape)


    ds = xr.Dataset(

    data_vars=dict(

        # test=(['lev' , 'ncol', 'nvertex'], np.zeros((lev, ncol, nvertex))),
        ptend_t=(['time_counter', 'lev' , 'ncol'], ptend_t),
        # area=(['ncol'], area_),
        ptend_q0001=(['time_counter', 'lev' , 'ncol'], ptend_q0001),
        cam_out_NETSW=(['time_counter', 'ncol'], cam_out_NETSW),
        cam_out_FLWDS=(['time_counter', 'ncol'], cam_out_FLWDS),
        cam_out_PRECSC=(['time_counter', 'ncol'], cam_out_PRECSC),
        cam_out_PRECC=(['time_counter', 'ncol'], cam_out_PRECC),
        cam_out_SOLS=(['time_counter', 'ncol'], cam_out_SOLS),
        cam_out_SOLL=(['time_counter', 'ncol'], cam_out_SOLL),
        cam_out_SOLSD=(['time_counter', 'ncol'], cam_out_SOLSD),
        cam_out_SOLLD=(['time_counter', 'ncol'], cam_out_SOLLD),

    
    ),

    coords=dict(

        lev=("lev", np.arange(0, 60, dtype=float)),
        # lat=("ncol", lat_), 
        # lon=("ncol", lon_),
        bounds_lat = (("ncol", "nvertex") , bounds_lat_),
        bounds_lon = (("ncol", "nvertex"), bounds_lon_),
        time_counter=("time_counter", range(T)),
        ncol=("ncol", range(ncol))
        



    )

    )
    

    

    for var in ds.variables:
        if str(var) == 'area': continue
        if str(var) == 'ncol': continue 
        if str(var) == 'lat' : ds[str(var)].attrs = {"coordinates": "bounds_lat"}
        elif str(var) == 'lon': ds[str(var)].attrs = {"coordinates": "bounds_lon"}
        elif (ds[str(var)].shape) == (ncol, ):
            print(str(var), ds[str(var)].shape)
            print(ds[str(var)].attrs)
            ds[str(var)].attrs = {"coordinates": "lon lat"}
            print(ds[str(var)].attrs)
        elif (ds[str(var)].shape) == (ncol, nvertex) : ds[str(var)].attrs = {"coordinates": "lon lat"}
        elif (ds[str(var)].shape) == (T, ncol) : ds[str(var)].attrs = {"coordinates": "time_centered lon lat"}
        elif (ds[str(var)].shape) == (T, lev, ncol) : ds[str(var)].attrs = {"coordinates": "time_centered lev lon lat"}


    # print('area coord ', ds['area'].attrs) 

    ds = xr.merge([grid, ds])

    return ds


