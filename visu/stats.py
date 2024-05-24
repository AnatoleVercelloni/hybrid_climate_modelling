import xarray as xr
import psyplot.project as psy
import psyplot
import numpy as np
import sys

grav    = 9.80616    # acceleration of gravity ~ m/s^2
cp      = 1.00464e3  # specific heat of dry air   ~ J/kg/K
lv      = 2.501e6    # latent heat of evaporation ~ J/kg
lf      = 3.337e5    # latent heat of fusion      ~ J/kg
ls      = lv + lf    # latent heat of sublimation ~ J/kg
rho_air = 101325./ (6.02214e26*1.38065e-23/28.966) / 273.15 # density of dry air at STP  ~ kg/m^3
                                                            # ~ 1.2923182846924677
                                                            # SHR_CONST_PSTD/(SHR_CONST_RDAIR*SHR_CONST_TKFRZ)
                                                            # SHR_CONST_RDAIR   = SHR_CONST_RGAS/SHR_CONST_MWDAIR
                                                            # SHR_CONST_RGAS    = SHR_CONST_AVOGAD*SHR_CONST_BOLTZ
rho_h20 = 1.e3       # density of fresh water     ~ kg/m^ 3


def load_ncfiles(filelist, grid_b):

    grid = psyplot.open_dataset(grid_b)

    print(grid)

    ncol = len(grid['ncol'])
    lev = len(grid['lev'])
    T = len(filelist)
    print('time counter = ', T)
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


def npy_toxarray(file, grid_b, unscale = "", var = 'ptend_t'):

    nvar = 128
    VAR = {'ptend_t': range(60), 'ptend_q0001': range(60,120,1), 'cam_out_NETSW': [120], 'cam_out_FLWDS': [121],
        'cam_out_PRECSC': [122], 'cam_out_PRECC': [123], 'cam_out_SOLS': [124], 'cam_out_SOLL': [125], 'cam_out_SOLSD': [126],
        'cam_out_SOLLD': [127], 'stat': range(60)}

    if var not in VAR:
        print("the variable is not available !")
        sys.exit(0)

    if type(file[0]) is str:
        DATA = []
        for i in range(len(file)):  
            data = np.load(file[i], mmap_mode='r')
            DATA.append(data)

        s = len(DATA[0])*len(file)
        data = np.array(DATA).reshape((s, nvar))

    else:
        if file.shape[1] == len(VAR[var]):
            data = file
            print("numpy array")
        else:
            print("wrong file parameter!")
            return 
    grid = psyplot.open_dataset(grid_b)

 
    ncol = len(grid['ncol'])
    T = data.shape[0]//ncol
    print('time counter = ', T)
    lev = len(grid['lev'])
    nvertex = len(grid['nvertex'])

    area_ = np.array(grid['area'])
    lon_ = np.array(grid['lon'])
    lat_ = np.array(grid['lat'])
    bounds_lon_ = np.array(grid['bounds_lon'])
    bounds_lat_ = np.array(grid['bounds_lat'])



    if 'stat'    in var: stat = data.reshape((T, ncol, lev)).transpose((0,2,1))
    if 'ptend_t' in var: ptend_t = data[:,:60].reshape((T,lev, ncol))
    if 'ptend_q0001' in var: ptend_q0001 = data[:,60:120].reshape((T, lev, ncol))
    if 'cam_out_NETSW' in var: cam_out_NETSW = data[:,120:121].reshape((T, ncol))
    if 'cam_out_FLWDS' in var: cam_out_FLWDS = data[:,121:122].reshape((T, ncol))
    if 'cam_out_PRECSC' in var: cam_out_PRECSC = data[:,122:123].reshape((T, ncol))
    if 'cam_out_PRECC' in var: cam_out_PRECC = data[:,123:124].reshape((T, ncol))
    if 'cam_out_SOLS' in var: cam_out_SOLS = data[:,124:125].reshape((T, ncol))
    if 'cam_out_SOLL' in var: cam_out_SOLL = data[:,125:126].reshape((T, ncol))
    if 'cam_out_SOLSD' in var: cam_out_SOLSD = data[:,126:127].reshape((T, ncol))
    if 'cam_out_SOLLD' in var: cam_out_SOLLD = data[:,127:128].reshape((T, ncol))

    if unscale != "":
        sc = xr.open_dataset(unscale)
        if 'ptend_t' in var: ptend_t = ptend_t/np.array(sc['ptend_t']).reshape((1,lev,1))
        if 'ptend_q0001' in var: ptend_q0001 = ptend_q0001/np.array(sc['ptend_q0001']).reshape((1,lev,1))
        if 'cam_out_NETSW' in var: cam_out_NETSW = cam_out_NETSW/np.array(sc['cam_out_NETSW']).reshape((1,1))
        if 'cam_out_FLWDS' in var: cam_out_FLWDS = cam_out_FLWDS/np.array(sc['cam_out_FLWDS']).reshape((1,1))
        if 'cam_out_PRECSC' in var: cam_out_PRECSC = cam_out_PRECSC/np.array(sc['cam_out_PRECSC']).reshape((1,1))
        if 'cam_out_PRECC' in var: cam_out_PRECC = cam_out_PRECC/np.array(sc['cam_out_PRECC']).reshape((1,1))
        if 'cam_out_SOLS' in var: cam_out_SOLS = cam_out_SOLS/np.array(sc['cam_out_SOLS']).reshape((1,1))
        if 'cam_out_SOLL' == var: cam_out_SOLL = cam_out_SOLL/np.array(sc['cam_out_SOLL']).reshape((1,1))
        if 'cam_out_SOLSD' in var: cam_out_SOLSD = cam_out_SOLSD/np.array(sc['cam_out_SOLSD']).reshape((1,1))
        if 'cam_out_SOLLD' in var: cam_out_SOLLD = cam_out_SOLLD/np.array(sc['cam_out_SOLLD']).reshape((1,1))

    # print(t.shape)


    ds = xr.Dataset(

    data_vars=dict(

        # if 'ptend_t' in var: ptend_t=(['time_counter', 'lev' , 'ncol'], ptend_t),
        # if 'ptend_q0001' in var: ptend_q0001=(['time_counter', 'lev' , 'ncol'], ptend_q0001),
        # if 'cam_out_NETSW' in var: cam_out_NETSW=(['time_counter', 'ncol'], cam_out_NETSW),
        # if 'cam_out_FLWDS' in var: cam_out_FLWDS=(['time_counter', 'ncol'], cam_out_FLWDS),
        # if 'cam_out_PRECSC' in var: cam_out_PRECSC=(['time_counter', 'ncol'], cam_out_PRECSC),
        # if 'cam_out_PRECC' in var: cam_out_PRECC=(['time_counter', 'ncol'], cam_out_PRECC),
        # if 'cam_out_SOLS' in var: cam_out_SOLS=(['time_counter', 'ncol'], cam_out_SOLS),
        # if 'cam_out_SOLL' in var: cam_out_SOLL=(['time_counter', 'ncol'], cam_out_SOLL),
        # if 'cam_out_SOLSD' in var: cam_out_SOLSD=(['time_counter', 'ncol'], cam_out_SOLSD),
        # if 'cam_out_SOLLD' in var: cam_out_SOLLD=(['time_counter', 'ncol'], cam_out_SOLLD),

    
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

    # print(cam_out_SOLS.shape)
    # print(ptend_t.shape)

    if 'stat'    in var: ds['stat']   = (['time_counter', 'lev', 'ncol'], stat)
    if 'ptend_t' in var: ds['ptend_t']=(['time_counter', 'lev' , 'ncol'], ptend_t)
    # if 'ptend_q0001' in var: ptend_q0001=(['time_counter', 'lev' , 'ncol'], ptend_q0001),
    # if 'cam_out_NETSW' in var: cam_out_NETSW=(['time_counter', 'ncol'], cam_out_NETSW),
    # if 'cam_out_FLWDS' in var: cam_out_FLWDS=(['time_counter', 'ncol'], cam_out_FLWDS),
    # if 'cam_out_PRECSC' in var: cam_out_PRECSC=(['time_counter', 'ncol'], cam_out_PRECSC),
    # if 'cam_out_PRECC' in var: cam_out_PRECC=(['time_counter', 'ncol'], cam_out_PRECC),
    if 'cam_out_SOLS' in var: ds['cam_out_SOLS']=(['time_counter', 'ncol'], cam_out_SOLS)
    # if 'cam_out_SOLL' in var: cam_out_SOLL=(['time_counter', 'ncol'], cam_out_SOLL),
    # if 'cam_out_SOLSD' in var: cam_out_SOLSD=(['time_counter', 'ncol'], cam_out_SOLSD),
    # if 'cam_out_SOLLD' in var: cam_out_SOLLD=(['time_counter', 'ncol'], cam_out_SOLLD),


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
        # if str(var) == 'stat': ds[str(var)].attrs = {"coordinates": "time_centered lev lon lat"}


    # print('area coord ', ds['area'].attrs) 

    ds = xr.merge([grid, ds])

    return ds


def compute_dp(grid):
    dp = np.array([grid['PI'][i+1] - grid['PI'][i] for i in range(60)])
    return dp, np.array(grid['area'])

def convert(data, dp):
    a = np.array(data['area'])
    pt = np.array(data['ptend_t'])
    p = a*dp*pt*cp/grav
    data['ptend_t_c'] = (("time_counter", "lev", "ncol"), p)
    data['ptend_t_c'].attrs = {"coordinates": "time_centered lev lon lat"}

    return 

# def mae():
#     (np.abs(DS_ENERGY['true']   - DS_ENERGY['pred'])).mean('time')




def compute_stats(filelist_true, filelist_pred, reso = 'low', var = 'ptend_t', unscale = "", conv = []):
    "compute mean, mae, r2. assume all files having the same length"

    D = dict()
    data = np.load(filelist_true[0], mmap_mode='r')
    lev = 60
    if reso == 'low': ncol = 384
    elif reso == 'high': ncol = 21600
    else:
        print("pick a valid resolution !")
        sys.exit(0)


    mae = np.zeros((lev))
    m2_ncol = np.zeros((ncol, lev))
    v_ncol = []
    mean_true_ncol = np.zeros((ncol, lev))
    mean_pred_ncol = np.zeros((ncol, lev))

    if data.shape[0]%(ncol) == 0:
        T = data.shape[0]//(ncol)
    else:
        print("dimension problem ! ")
        return 

    #mean_area = (np.array(conv[1]).sum()/ncol)
   
    if unscale != "" : sc = xr.open_dataset(unscale)
    
    if var == 'ptend_t':
        idx = range(60)
    else:
        print("variable not available !")
        return

    for (file_t, file_p) in zip(filelist_true, filelist_pred):

        # print("loading "+ file + " ncol = "+str(ncol)+", T = "+str(T))
        data_t = np.load(file_t, mmap_mode='r')
        data_t = data_t[:,idx].reshape((ncol, T, 60))
        if unscale != "": data_t = data_t/np.array(sc['ptend_t']).reshape((1,1,lev))
        if len(conv) != 0: data_t = (conv[1].reshape(ncol, 1, 1))*(conv[0].reshape(ncol, 1, lev))*data_t*cp/(grav*mean_area)

        data_p = np.load(file_p, mmap_mode='r')
        data_p = data_p[:,:60].reshape((ncol, T, 60))
        if unscale != "": data_p = data_p/np.array(sc['ptend_t']).reshape((1,1,lev))
        if len(conv) != 0: data_p = (conv[1].reshape(ncol, 1, 1))*(conv[0].reshape(ncol, 1, lev))*data_p*cp/(grav*mean_area)

        mean_true_ncol = mean_true_ncol + (np.abs(data_t)).sum(axis=1)
        mean_pred_ncol = mean_pred_ncol + (np.abs(data_p)).sum(axis=1)

        mae = mae + (np.abs(data_t - data_p)).sum(axis=(0,1))

        m2_ncol = m2_ncol + ((data_t - data_p)**2).sum(axis=1)
        v_ncol.append(data_t.reshape(ncol,T,lev))

    print("stats computed on "+str(T*len(filelist_pred))+" time steps")
    D['mean_true_ncol'] = mean_true_ncol/(T*len(filelist_true))
    D['mean_pred_ncol'] = mean_pred_ncol/(T*len(filelist_true))

    D['mean_true'] = D['mean_true_ncol'].sum(axis = 0)/(ncol)
    D['mean_pred'] = D['mean_pred_ncol'].sum(axis = 0)/(ncol)

    D['mae'] = mae/(T*len(filelist_true)*ncol)

    v_ncol = (((np.array(v_ncol).reshape(len(filelist_pred), ncol, T, lev)) - (D["mean_true_ncol"].reshape(1, ncol, 1, lev)))**2).sum(axis=(0,2))
    D['r2_ncol'] = 1 - m2_ncol/v_ncol

    D['r2'] = D['r2_ncol'].sum(axis = 0)/ncol

    return D

    

        
