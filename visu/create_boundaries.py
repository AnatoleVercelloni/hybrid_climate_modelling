import netCDF4
from netCDF4 import Dataset
from plotClimsim import mesh, format_, matplotlib_plot
import numpy as np

def read_grid_info(filename, verbose = True):
    nc = netCDF4.Dataset(filename)
    # print(filename)
    
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]
    P0 = nc.variables['P0'][:]
    PS = nc.variables['PS'][0][:]
    area = nc.variables['area'][:]
    hyai = nc.variables['hyai'][:]
    hyam = nc.variables['hyam'][:]
    hybi = nc.variables['hybi'][:]
    hybm = nc.variables['hybm'][:]

    nvertex = 10

    PI = np.outer(hyai*P0, np.ones(384)) + np.outer(hybi,PS)

    

    ncel = lat.shape[0]

    POLY, nvertex = mesh(filename, nvertex, verbose)

    print("read grid info from netcedf file ok !")
    

    return ncel, POLY, nvertex, lon, lat, P0, PS, PI, area, hyai, hyam, hybi, hybm 

def read_variable(filename, variable):

    print("reading "+ variable + " from netcedf file"+ filename)

    nc = netCDF4.Dataset(filename)
    var = nc.variables[variable][:]



    print("reading   ok !")

    return  var



def create_grid_info(filename, grid_info, verbose=True):


    ncel, POLY, nvertex, lon_, lat_, P0_, PS_, PI_, area_, hyai_, hyam_, hybi_, hybm_ = read_grid_info(grid_info, verbose)

    # print(lat_)

    bounds_lon_, bounds_lat_ = format_(POLY)


    rootgrp = Dataset(filename, "w", format="NETCDF4")

    
    nb = rootgrp.createDimension("axis_nbounds", 2)
    ilev = rootgrp.createDimension("ilev", 61)
    lev = rootgrp.createDimension("lev", 60)
    ncol = rootgrp.createDimension("ncol", ncel)
    nvertex = rootgrp.createDimension("nvertex", nvertex)
    

	
    P0 = rootgrp.createVariable("P0","f8",())
    PS = rootgrp.createVariable("Ps","f8",())
    PI = rootgrp.createVariable("PI","f8",("ilev", "ncol", ))

    area = rootgrp.createVariable("area","f8",("ncol",))
    hyai = rootgrp.createVariable("hyai","f8",("ilev",))
    hyam = rootgrp.createVariable("hyam","f8",("lev",))
    hybi = rootgrp.createVariable("hybi","f8",("ilev",))
    hybm = rootgrp.createVariable("hybm","f8",("lev",))
    lat = rootgrp.createVariable("lat","f8",("ncol",))
    lon = rootgrp.createVariable("lon","f8",("ncol",))
    lev = rootgrp.createVariable("lev", "f8", ("lev",))
    ilev = rootgrp.createVariable("ilev", "f8", ("ilev",))

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


    hyai.long_name = 'hybrid A coefficient at layer interfaces'

    hyam.long_name = 'hybrid A coefficient at layer midpoints'

    hybi.long_name = 'hybrid B coefficient at layer interfaces'

    hyai.long_name = 'hybrid B coefficient at layer midpoints'

    lev.long_name = 'hybrid level at midpoints (1000*(A+B))'
    lev.units = 'hPa' 
    lev.positive = 'down' 
    lev.standard_name = 'atmosphere_hybrid_sigma_pressure_coordinate'
    lev.formula_terms = 'a: hyam b: hybm p0: P0 ps: PS'


    rootgrp.description = "test_climsim"

    lev[:] = range(60)
    ilev[:] = range(61)
    lat[:] = lat_
    lon[:] = lon_



    bounds_lat[:] = bounds_lat_
    bounds_lon[:] = bounds_lon_

    # print(rootgrp['bounds_lon'])
    
    area[:] = area_
    P0 = P0_
    PS = PS_

    PI[:] = PI_
    hyai[:] = hyai_
    hyam[:] = hyam_
    hybi[:] = hybi_
    hybm[:] = hybm_
	
    rootgrp.close()

    return rootgrp


