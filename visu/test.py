from stats import *
import psyplot.project as psy
from create_boundaries import create_grid_info

reso = "low"

data_path = "/gpfsdswork/dataset/ClimSim_low-res/train/"
filelist = [data_path + '0008-02/E3SM-MMF.mli.0008-02-**-00000.nc']
grid_info = '/gpfswork/rech/psl/upu87pm/ClimSim/grid_info/ClimSim_low-res_grid-info.nc'

grid_b = '../data/nc_data/grid_with_boundaries_'+reso + 'res.nc'
nc_file = 'test.nc'

var = 'area'
create_grid_info(grid_b, grid_info)

# # print(grid_b)
# In, Out = load_ncfiles(filelist, grid_b)

# In.to_netcdf(nc_file)


# # m_t = map_meant_var(In, var)



# tmap = psy.plot.mapplot(nc_file, name = var)
# tmap.save_project("psyplot_test.png")
                


