import netCDF4
from netCDF4 import Dataset
from plotClimsim import mesh, format_, matplotlib_plot
import numpy as np
import xarray as xr
import os

def read_grid_info(filename, verbose = True):
    nc = netCDF4.Dataset(filename)
    lat = []
    
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]
    P0 = nc.variables['P0'][:]
    area = nc.variables['area'][:]

    #bounds_lat = nc.variables['bounds_lat'][:]
    nvertex = 10

    

    ncel = lat.shape[0]

    POLY, nvertex = mesh(filename, nvertex, verbose)

    print("read grid info from netcedf file ok !")

    return ncel, POLY, nvertex, lon, lat, P0, area

def read_variable(filename, variable):

    print("reading "+ variable + " from netcedf file"+ filename)

    nc = netCDF4.Dataset(filename)
    var = nc.variables[variable][:]



    print("reading   ok !")

    return  var



def create_grid_info(filename, grid_info, verbose=True):


    ncel, POLY, nvertex, lon_, lat_, P0_, area_ = read_grid_info(grid_info, verbose)

    # print(lat_)

    bounds_lon_, bounds_lat_ = format_(POLY)


    rootgrp = Dataset(filename, "w", format="NETCDF4")

    
    nb = rootgrp.createDimension("axis_nbounds", 2)
    ilev = rootgrp.createDimension("ilev", 61)
    lev = rootgrp.createDimension("lev", 60)
    ncol = rootgrp.createDimension("ncol", ncel)
    nvertex = rootgrp.createDimension("nvertex", nvertex)
    

	
    P0 = rootgrp.createVariable("P0","f8",())
    area = rootgrp.createVariable("area","f8",("ncol",))
    # hyai = rootgrp.createVariable("hyai","f8",("ilev",))
    # hyam = rootgrp.createVariable("hyam","f8",("lev",))
    # hybi = rootgrp.createVariable("hybi","f8",("ilev",))
    # hybm = rootgrp.createVariable("hybm","f8",("lev",))
    lat = rootgrp.createVariable("lat","f8",("ncol",))
    lon = rootgrp.createVariable("lon","f8",("ncol",))
    lev = rootgrp.createVariable("lev", "f8", ("lev",))
    bounds_lat = rootgrp.createVariable("bounds_lat","f8",("ncol","nvertex",))
    bounds_lon = rootgrp.createVariable("bounds_lon","f8",("ncol","nvertex",))

   


    
    lat.long_name = 'latitude'
    lat.units = 'degrees_north'
    lat.bounds = 'bounds_lat'

    lon.long_name = 'longitude'
    lon.units = 'degrees_east'
    lon.bounds = 'bounds_lon'

    P0.long_name = ' reference pressure'
    P0.units = 'Pa'

    area.long_name = 'physics grid areas'
    area.coordinates = "lon lat"

    # lat.coordinates = 'lon lat'
    # lon.coordinates = 'lon lat'
    print('add coordinates into  '+ filename)
    # hyai.long_name = 'hybrid A coefficient at layer interfaces'

    # hyam.long_name = 'hybrid A coefficient at layer midpoints'

    # hybi.long_name = 'hybrid B coefficient at layer interfaces'

    # hyai.long_name = 'hybrid B coefficient at layer midpoints'

    lev.long_name = 'hybrid level at midpoints (1000*(A+B))'
    lev.units = 'hPa' 
    lev.positive = 'down' 
    lev.standard_name = 'atmosphere_hybrid_sigma_pressure_coordinate'
    lev.formula_terms = 'a: hyam b: hybm p0: P0 ps: PS'


    rootgrp.description = "test_climsim"

    lev[:] = range(60)
    # ilev[:] = range(61)
    lat[:] = lat_
    lon[:] = lon_



    bounds_lat[:] = bounds_lat_
    bounds_lon[:] = bounds_lon_

    # print(rootgrp['bounds_lon'])
    
    area[:] = area_
    P0 = P0_
	
    rootgrp.close()

    return rootgrp

def create_visu_input(grid, filename, newfile):
    grid_ = netCDF4.Dataset(grid)
    data_ = netCDF4.Dataset(filename)

    

    
    ncol = grid_['area'].shape[0]
    # print(ncol)
    # print(filename)
    if ncol != data_['state_ps'].shape[0]:
        print("error, the resolution is different for the grid and the data !")
        return 
    
    nc_merged = xr.open_mfdataset([grid, filename])


    for var in nc_merged.variables:
        # print((nc_merged[var].shape))
        #del((nc_merged[var].attrs)['_FillValue'])
        if nc_merged[var].shape == (ncol,):
            (nc_merged[var].attrs)['coordinates'] = 'lon lat'
        elif nc_merged[var].shape == (61,):
            (nc_merged[var].attrs)['coordinates'] = 'ilev'
        elif nc_merged[var].shape == (60,):
            (nc_merged[var].attrs)['coordinates'] =  'lev'
        elif nc_merged[var].shape == (60, ncol):
            (nc_merged[var].attrs)['coordinates'] = 'lon lat lev'


    nc_merged.to_netcdf(newfile)

    # nvertex = nc_merged.createDimension("prediction", 2)

    grid_.close()
    data_.close()
    nc_merged.close()


    return




def create_visu_output(grid, filename, newfile):
    vars_mli = [ 'state_v', 'state_pmid', 'pbuf_TAUY', 'pbuf_TAUX', 'pbuf_SOLIN', 'pbuf_SHFLX', 'pbuf_LHFLX', 'pbuf_COSZRS', 'cam_in_SNOWHLAND', 'cam_in_SNOWHICE', 'cam_in_OCNFRAC', 'cam_in_LWUP', 'cam_in_LANDFRAC', 'cam_in_ICEFRAC', 'cam_in_ALDIF', 'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR','state_t','state_q0001', 'state_q0002', 'state_q0003', 'state_u', 'state_ps','pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
    vars_mlo = ['state_t', 'state_u', 'state_v', 'ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC','cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']

    ds = xr.open_dataset(filename.replace('.mlo.','.mli.'), engine='netcdf4')
    # print(filename, ds)
    ds = ds[vars_mli]                             
            
    # read mlo
    dso = xr.open_dataset(filename, engine='netcdf4') 
            
    # make mlo variales: ptend_t and ptend_q0001
    dso['ptend_t'] = (dso['state_t'] - ds['state_t'])/1200 # T tendency [K/s]
    dso['ptend_q0001'] = (dso['state_q0001'] - ds['state_q0001'])/1200 # Q tendency [kg/kg/s]
    dso = dso[vars_mlo]

    # print(dso)/

    dso.to_netcdf('tmp.nc')

    grid_ = netCDF4.Dataset(grid)

    data_ = netCDF4.Dataset('tmp.nc')
   
    
    ncol = grid_['area'].shape[0]
    # print(filename)
    if ncol != data_['cam_out_SOLLD'].shape[0]:
        print("error, the resolution is different for the grid and the data !")
        return 
    
    nc_merged = xr.open_mfdataset([grid,'tmp.nc'])


    for var in nc_merged.variables:
        # print((nc_merged[var].shape))
        #del((nc_merged[var].attrs)['_FillValue'])
        if nc_merged[var].shape == (ncol,):
            (nc_merged[var].attrs)['coordinates'] = 'lon lat'
        elif nc_merged[var].shape == (61,):
            (nc_merged[var].attrs)['coordinates'] = 'ilev'
        elif nc_merged[var].shape == (60,):
            (nc_merged[var].attrs)['coordinates'] =  'lev'
        elif nc_merged[var].shape == (60, ncol):
            (nc_merged[var].attrs)['coordinates'] = 'lon lat lev'


    nc_merged.to_netcdf(newfile)

    # n = netCDF4.Dataset(newfile, "a" )
    # d = n.createDimension("prediction", 2)
    # n.close()

    grid_.close()
    data_.close()
    nc_merged.close()


    return
    
        
        



def plot_ncdf(grid_info, fn_in, fn_out, var, Ai=False, matplotlib=True):
    ncel, POLY, nvertex, lon, lat = read_grid_info(grid_info, Ai)
    variable = read_variable(fn_in, var)
    #print("ncel", ncel)
    bounds_lon, bounds_lat = format_(POLY)
    #root = createnetcdf(fn_out, ncel, nvertex, lat, lon, bounds_lat, bounds_lon, variable, True)
    if matplotlib:
        matplotlib_plot(POLY, ['0' if variable[i]>0.5 else '1' for i in range (len(variable))])


def npy_to_netcdf_inputv1(filename, grid_info_b, new_netcdf_file, reso='low'):


    tmp_file = "data/tmp.nc"

    data = np.load(filename)


    rootgrp = Dataset(new_netcdf_file, "w", format="NETCDF4")

    # print(data[:,:2])

    ncel = 384 if reso=='low' else 21600

    nc = netCDF4.Dataset(grid_info_b)

    lat_ = nc.variables['lat'][:]
    lon_ = nc.variables['lon'][:]
    bounds_lat_ = nc.variables['bounds_lat'][:]
    bounds_lon_ = nc.variables['bounds_lon'][:]
    lev_ = nc.variables['lev'][:]
 
    P0_ = nc.variables['P0'][:]
    area_ = nc.variables['area'][:]

    #bounds_lat = nc.variables['bounds_lat'][:]
    nvertex = 10


    ncel = lat_.shape[0]



    nb = rootgrp.createDimension("axis_nbounds", 2)
    ilev = rootgrp.createDimension("ilev", 61)
    lev = rootgrp.createDimension("lev", 60)
    cel = rootgrp.createDimension("ncol", ncel)
    nvertex = rootgrp.createDimension("nvertex", nvertex)

    P0 = rootgrp.createVariable("P0","f8",())
    area = rootgrp.createVariable("area","f8",("ncol",))
    lat = rootgrp.createVariable("lat","f8",("ncol",))
    lon = rootgrp.createVariable("lon","f8",("ncol",))
    lev = rootgrp.createVariable("lev","f8",("lev",))

    bounds_lat = rootgrp.createVariable("bounds_lat","f8",("ncol","nvertex",))
    bounds_lon = rootgrp.createVariable("bounds_lon","f8",("ncol","nvertex",))

    
	
    state_q0001 = rootgrp.createVariable("state_q0001", "f8", ("lev" , "ncol",))
    state_t = rootgrp.createVariable("state_t", "f8", ("lev" , "ncol",))
    state_ps = rootgrp.createVariable("state_ps", "f8", ( "ncol",))
    pbuf_SOLIN = rootgrp.createVariable("pbuf_SOLIN", "f8", ("ncol",))
    pbuf_LHFLX = rootgrp.createVariable("pbuf_LHFLX", "f8", ("ncol",))
    pbuf_SHFLX = rootgrp.createVariable("pbuf_SHFLX", "f8", ("ncol",))

    area.coordinates = "lon lat"
    
    state_q0001.coordinates = "lon lat lev"
    state_t.coordinates = "lon lat lev"
    state_ps.coordinates = "lon lat"
    pbuf_SOLIN.coordinates = "lon lat"
    pbuf_LHFLX.coordinates = "lon lat"
    pbuf_SHFLX.coordinates = "lon lat"
    lon.coordinates = "lon lat"
    lat.bounds = "bounds_lat"
    lon.bounds = "bounds_lon"
    lat.coordinates = "lon lat"
    lev.coordinates = "lev"


    lat[:] = lat_
    lon[:] = lon_

    bounds_lat[:] = bounds_lat_
    bounds_lon[:] = bounds_lon_
    lev[:] = lev_
    area[:] = area_
    P0 = P0_
	

    state_t[:] = data[:,:60].T
    state_q0001[:] = data[:,60:120].T
    state_ps[:] = data[:,120:121].T
    pbuf_SOLIN[:] = data[:,121:122].T
    pbuf_LHFLX[:] = data[:,122:123].T
    pbuf_SHFLX[:] = data[:,123:124].T



    rootgrp.description = "input_ClimSim"

 
	
    rootgrp.close()
    nc.close()


    return 

def npy_to_netcdf_outputv1(filename, grid_info_b, new_netcdf_file, reso='low'):


    tmp_file = "data/tmp.nc"
    data = np.load(filename)
    rootgrp = Dataset(new_netcdf_file, "w", format="NETCDF4")


    ncel = 384 if reso=='low' else 21600
    nc = netCDF4.Dataset(grid_info_b)

    lat_ = nc.variables['lat'][:]
    lon_ = nc.variables['lon'][:]
    bounds_lat_ = nc.variables['bounds_lat'][:]
    bounds_lon_ = nc.variables['bounds_lon'][:]
    lev_ = nc.variables['lev'][:]
 
    P0_ = nc.variables['P0'][:]
    area_ = nc.variables['area'][:]
    nvertex = 10


    ncel = lat_.shape[0]



    nb = rootgrp.createDimension("axis_nbounds", 2)
    ilev = rootgrp.createDimension("ilev", 61)
    lev = rootgrp.createDimension("lev", 60)
    cel = rootgrp.createDimension("ncol", ncel)
    nvertex = rootgrp.createDimension("nvertex", nvertex)
    # prediction = rootgrp.createDimension("prediction", 2)

    P0 = rootgrp.createVariable("P0","f8",())
    area = rootgrp.createVariable("area","f8",("ncol",))
    lat = rootgrp.createVariable("lat","f8",("ncol",))
    lon = rootgrp.createVariable("lon","f8",("ncol",))
    bounds_lat = rootgrp.createVariable("bounds_lat","f8",("ncol","nvertex",))
    bounds_lon = rootgrp.createVariable("bounds_lon","f8",("ncol","nvertex",))

    
	
    ptend_q0001 = rootgrp.createVariable("ptend_q0001_pred", "f8", ("lev" , "ncol",))
    ptend_t = rootgrp.createVariable("ptend_t_pred", "f8", ( "lev" , "ncol",))
    cam_out_NETSW = rootgrp.createVariable("cam_out_NETSW_pred", "f8", ( "ncol",))
    cam_out_FLWDS = rootgrp.createVariable("cam_out_FLWDS_pred", "f8", ("ncol",))
    cam_out_PRECSC = rootgrp.createVariable("cam_out_PRECSC_pred", "f8", ("ncol",))
    cam_out_PRECC = rootgrp.createVariable("cam_out_PRECC_pred", "f8", ("ncol",))
    cam_out_SOLS = rootgrp.createVariable("cam_out_SOLS_pred", "f8", ("ncol",))
    cam_out_SOLL = rootgrp.createVariable("cam_out_SOLL_pred", "f8", ("ncol",))
    cam_out_SOLSD = rootgrp.createVariable("cam_out_SOLSD_pred", "f8", ("ncol",))
    cam_out_SOLLD = rootgrp.createVariable("cam_out_SOLLD_pred", "f8", ("ncol",))
    levv = rootgrp.createVariable("lev", "f8", ("lev",))


    lon.coordinates = "lon lat"
    lat.coordinates = "lon lat"
    area.coordinates = "lon lat"
    levv.coordinates = "lev"
    
    ptend_q0001.coordinates = "lon lat lev"
    ptend_t.coordinates = "lon lat lev"
    cam_out_NETSW.coordinates = "lon lat"
    cam_out_FLWDS.coordinates = "lon lat"
    cam_out_PRECSC.coordinates = "lon lat"
    cam_out_PRECC.coordinates = "lon lat"
    cam_out_SOLS.coordinates = "lon lat"
    cam_out_SOLL.coordinates = "lon lat"
    cam_out_SOLSD.coordinates = "lon lat"
    cam_out_SOLLD.coordinates = "lon lat"
    lat.bounds = "bounds_lat"
    lon.bounds = "bounds_lon"

    levv[:] = lev_
    lat[:] = lat_
    lon[:] = lon_

    bounds_lat[:] = bounds_lat_
    bounds_lon[:] = bounds_lon_
    
    area[:] = area_
    P0 = P0_
	
    # print(nc.variables['area'].shape)


    ptend_t[:] = data[:,:60].T
    ptend_q0001[:] = data[:,60:120].T
    cam_out_NETSW[:] = data[:,120:121].T
    cam_out_FLWDS[:] = data[:,121:122].T
    cam_out_PRECSC[:] = data[:,122:123].T
    cam_out_PRECC[:] = data[:,123:124].T
    cam_out_SOLS[:] = data[:,124:125].T
    cam_out_SOLL[:] = data[:,125:126].T
    cam_out_SOLSD[:] = data[:,126:127].T
    cam_out_SOLLD[:] = data[:,127:128].T


    rootgrp.description = "output_ClimSim"

    # print("hello")
	
    rootgrp.close()
    nc.close()


    return 












def add_prediction(true, pred, merge, scale = ""):
    """ take a true output file and a prediction file, unscale prediction and merge them into one file"""

    
    
    nc_merged = xr.open_mfdataset([true, pred])

    if scale != "":
        scale_ = xr.open_dataset(scale)
        for var in nc_merged.variables:
            # print(var[-5:])
            if var[-5:] == "_pred":
                print("unscalling")
                nc_merged[var] = nc_merged[var]/scale_[var[:-5]]
        scale_.close()

    nc_merged.to_netcdf(merge)

    nc_merged.close()
    

    return 
    