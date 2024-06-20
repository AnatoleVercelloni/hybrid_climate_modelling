import xarray as xr
import psyplot.project as psy
import psyplot
import numpy as np
import sys

# constants for unit conversion #

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
    #take a list of netcdf file (one file = one snapshot) of data and a grid info netcdf file as input
    #return the xarraydataset which merge all this data and add a time dimension and can be used by psyplot

    #open the grid and get the dimensions
    grid = psyplot.open_dataset(grid_b)
    ncol = len(grid['ncol'])
    lev = len(grid['lev'])
    T = len(filelist)
    print('time counter = ', T)
    nvert = len(grid['nvertex'])

    #open the data
    inputs = [xr.open_dataset(file) for file in filelist]

    #add the time dimension
    In = xr.concat(inputs, dim = 'time_counter')
    In['time_counter'] = range(T)

    #merge everything
    In = xr.merge([grid, In])

    #add the coordinates attribute for psyplot
    for var in In.variables:
        if str(var) == 'area': continue
        if (In[str(var)].shape) == (ncol, ): In[str(var)].attrs = {"coordinates": "lon lat"}
        elif (In[str(var)].shape) == (ncol, nvert) : In[str(var)].attrs = {"coordinates": "lon lat"}
        elif (In[str(var)].shape) == (T, ncol) : In[str(var)].attrs = {"coordinates": "time_centered lon lat"}
        elif (In[str(var)].shape) == (T, lev, ncol) : In[str(var)].attrs = {"coordinates": "time_centered lev lon lat"}


    #add the bounds attribute for psyplot
    In['lon'].attrs = {"bounds": "bounds_lon"}
    In['lat'].attrs = {"bounds": "bounds_lat"}


    #same for the output files 
    outputs = [xr.open_dataset(file.replace('.mli.','.mlo.')) for file in filelist]
    Ou = xr.concat(outputs, dim = 'time_counter')
    Ou['time_counter'] = range(T)

    Ou = xr.merge([grid, Ou])

    Ou['ptend_t'] = (Ou['state_t'] - In['state_t'])/1200
    Ou['ptend_q0001'] = (Ou['state_q0001'] - In['state_q0001'])/1200

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
    #take either a list of .npy file or a numpy array for 'file', a netcdf file of the grid info for 'grid_b' 
    #if unscale is set to a netcdf file with the unscale factor, it also unscale the data
    #var is the variable which is load
    #==> it create a xarraydataset for this data and that can be used by psyplot

    #open the grid and get the dimensions
    grid = psyplot.open_dataset(grid_b)
    ncol = len(grid['ncol'])
    lev = len(grid['lev'])
    nvertex = len(grid['nvertex'])

    #create a dictionnary of the available variables ('stat' is for any postprocessed data)
    VAR = {'ptend_t': range(lev), 'ptend_q0001': range(lev,2*lev,1), 'cam_out_NETSW': [120], 'cam_out_FLWDS': [121],
        'cam_out_PRECSC': [122], 'cam_out_PRECC': [123], 'cam_out_SOLS': [124], 'cam_out_SOLL': [125], 'cam_out_SOLSD': [126],
        'cam_out_SOLLD': [127], 'stat': range(lev)}
    

    #check var is set corectly 
    if var not in VAR:
        print("the variable is not available !")
        sys.exit(0)

    #get the length of the variable
    nvar = len(VAR[var])

    #check the input type (here .npy file list)
    if type(file[0]) is str:
        DATA = []

        #load just the data of var for each file
        for i in range(len(file)):  
            data = np.load(file[i], mmap_mode='r')
            data = data[:,VAR[var]]
            DATA.append(data)

        #shape the data to be (time*ncol)x(var)
        s = len(DATA[0])*len(file)
        data = np.array(DATA).reshape((s, nvar))

    #numpy array
    else:
        #check that the data is coherent with the variable
        if file.shape[1] == len(VAR[var]):
            data = file
        else:
            print("wrong file parameter!")
            return 

    T = data.shape[0]//ncol
    print('time counter = ', T)

    #reshape the data 
    if type(file[0]) is str:
        if nvar == lev:  data = data.reshape(T, lev, ncol)
        if nvar == 1: data = data.reshape(T, ncol)

    #The axis are inverted for stat    
    if 'stat' in var: data = data.reshape((T, ncol, lev)).transpose((0,2,1))

    #get the bpunds_lon(resp. lat) for the coordinates
    bounds_lon_ = np.array(grid['bounds_lon'])
    bounds_lat_ = np.array(grid['bounds_lat'])

    #if unscale is set, unscale the data
    if unscale != "" and 'stat' not in var:
        sc = xr.open_dataset(unscale)
        if nvar == lev: 
            data = data/np.array(sc[var]).reshape((1,lev,1))
        else:
            data = data/np.array(sc[var]).reshape((1,1))

    #We can now create a xarraydataset for our data
    ds = xr.Dataset(

        #add the dimensions that are coordinates (for psyplot)
        coords=dict(
            lev=("lev", np.arange(0, lev, dtype=float)),
            bounds_lat = (("ncol", "nvertex") , bounds_lat_),
            bounds_lon = (("ncol", "nvertex"), bounds_lon_),
            time_counter=("time_counter", range(T)),
            ncol=("ncol", range(ncol))
        )

    )


    #add the data to the dataset
    if nvar == lev: ds[var]   = (['time_counter', 'lev', 'ncol'], data)
    elif nvar == 1: ds[var]   = (['time_counter', 'ncol'], data)


    #also set the coordinate attribute (for psyplot), redundant but doesn't work if it's not done
    for var in ds.variables:
        if str(var) == 'area': continue
        if str(var) == 'ncol': continue 
        if str(var) == 'lat' : ds[str(var)].attrs = {"coordinates": "bounds_lat"}
        elif str(var) == 'lon': ds[str(var)].attrs = {"coordinates": "bounds_lon"}
        elif (ds[str(var)].shape) == (ncol, ):
            ds[str(var)].attrs = {"coordinates": "lon lat"}
        elif (ds[str(var)].shape) == (ncol, nvertex) : ds[str(var)].attrs = {"coordinates": "lon lat"}
        elif (ds[str(var)].shape) == (T, ncol) : ds[str(var)].attrs = {"coordinates": "time_centered lon lat"}
        elif (ds[str(var)].shape) == (T, lev, ncol) : ds[str(var)].attrs = {"coordinates": "time_centered lev lon lat"}


    #merge data with the grid
    ds = xr.merge([grid, ds])

    return ds


def compute_dp(grid):
    #take an xarraydataset grid info and return the list of deltap interface

    dp = np.array([grid['PI'][i+1] - grid['PI'][i] for i in range(60)])
    return dp, np.array(grid['area'])




def compute_stats(filelist_true, filelist_pred, reso = 'low', var = 'ptend_t', unscale = "", conv = []):
    #compute mean, mae, r2. assume all files(.npy) having the same length
    #assume that we don't need to compute stats of non vertical variables

    #create a dictionary of metrics
    D = dict()

    #open the first file to get the dimensions
    data = np.load(filelist_true[0], mmap_mode='r')
    data_p = np.load(filelist_pred[0], mmap_mode='r')

    lev = 60

    #create a dictionnary of the available variables ('stat' is for any postprocessed data)
    VAR = {'ptend_t': range(lev), 'ptend_q0001': range(lev,2*lev,1), 'ptend_q0001': range(2*lev,3*lev,1), 'ptend_q0001': range(3*lev,4*lev,1), 
           'ptend_u': range(4*lev,5*lev,1), 'ptend_v': range(5*lev,6*lev,1),
           'cam_out_NETSW': [120], 'cam_out_FLWDS': [121],
           'cam_out_PRECSC': [122], 'cam_out_PRECC': [123], 'cam_out_SOLS': [124], 'cam_out_SOLL': [125], 'cam_out_SOLSD': [126],
           'cam_out_SOLLD': [127], 'stat': range(lev)}

    #create a dictionary of conversion associated to the variables
    CONV = {'ptend_t': cp, 'ptend_q0001': lv}
    

    #check var is set corectly 
    if var not in VAR:
        print("the variable is not available !")
        sys.exit(0)

    #get the length of the variable
    nvar = len(VAR[var])


    #set the resolution
    if reso == 'low': ncol = 384
    elif reso == 'high': ncol = 21600
    else:
        print("pick a valid resolution !")
        sys.exit(0)

    #define the metrics
    mae = np.zeros((lev))
    m2_ncol = np.zeros((ncol, lev))
    v_ncol = []
    mean_true_ncol = np.zeros((ncol, lev))
    mean_pred_ncol = np.zeros((ncol, lev))

    #check the reolution is coherent
    if data.shape[0]%(ncol) == 0:
        T = data.shape[0]//(ncol)
    else:
        print("dimension problem ! ")
        return 
        
    if data.shape != data_p.shape:
        print("predicted data and target data have different shapes !")
        print("pred : ", data_p.shape, " != targ : ", data.shape)
        return 
    if len(filelist_true) != len(filelist_pred):
        print("you should have the same number of file for predicted and target data !")
        return 
        
    #if conversion set, need the mean_area
    if len(conv) != 0: mean_area = (np.array(conv[1]).sum()/ncol)
   
    if unscale != "" : sc = xr.open_dataset(unscale)

    #iterate over the file to limit memory issues
    for (file_t, file_p) in zip(filelist_true, filelist_pred):

        #open the file of true data, load the data of var and update the metrics
        data_t = np.load(file_t, mmap_mode='r')
        data_t = data_t[:,VAR[var]].reshape((ncol, T, lev))
        if unscale != "": data_t = data_t/np.array(sc[var]).reshape((1,1,lev))
        if len(conv) != 0: data_t = (conv[1].reshape(ncol, 1, 1))*(conv[0].reshape(ncol, 1, lev))*data_t*CONV[var]/(grav*mean_area)

        #do the same with predict data
        data_p = np.load(file_p, mmap_mode='r')
        data_p = data_p[:,VAR[var]].reshape((ncol, T, lev))
        if unscale != "": data_p = data_p/np.array(sc[var]).reshape((1,1,lev))
        if len(conv) != 0: data_p = (conv[1].reshape(ncol, 1, 1))*(conv[0].reshape(ncol, 1, lev))*data_p*CONV[var]/(grav*mean_area)

        mean_true_ncol = mean_true_ncol + (np.abs(data_t)).sum(axis=1)
        mean_pred_ncol = mean_pred_ncol + (np.abs(data_p)).sum(axis=1)

        mae = mae + (np.abs(data_t - data_p)).sum(axis=(0,1))

        m2_ncol = m2_ncol + ((data_t - data_p)**2).sum(axis=1)
        v_ncol.append(data_t.reshape(ncol,T,lev))

    #compute the metrics and fill in the dictionary
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

    

        
