from tonetcdf import read_variable
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os.path
import psyplot.project as psy
import psyplot
from psy_maps.plotters import FieldPlotter
from netCDF4 import Dataset
from tonetcdf import create_grid_info, create_visu_input, create_visu_output



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



# vars_mli = ['state_pmid','state_t','state_q0001','state_ps','pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
# vars_mlo = ['ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC','cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']



def tend(filename, var):

    ds = xr.open_dataset(filename, engine='netcdf4')
    
    ds = ds[vars_mli]   
                           
            
    # read mlo
    dso = xr.open_dataset(filename.replace('.mli.','.mlo.'), engine='netcdf4') 


    dso['ptend_'+var] = (dso['state_'+var] - ds['state_'+var])/1200 # T tendency [K/s]

    dso = dso['ptend_'+var]

    #print(ds.keys())

    return ds, dso



def plot_variable_vertical_K(filename, var, col):



    ds, dso = tend(filename, var)


    plt.plot(range(1,61), dso['ptend_'+str(var)][:,col] , label=str(var)+'+_out - '+str(var)+'-in')
    plt.xlabel('vertical level')
    plt.ylabel('variation de '+str(var))
    plt.legend()
    plt.show()


def plot_t_vertical_Wm2(filename, col):

    ds, dso = tend(filename, 't')
    dp = delta_p(ds, col)

    twm2 = [cp/grav*dp[i] for i in range(58)]
    
    plt.plot(range(1,59), twm2)

    plt.xlabel('vertical level')
    plt.ylabel('variation de temperature en w/m^2')
    plt.legend()
    plt.show()


def delta_p(ds, col):
        p = ds['state_pmid'][:,col]
        p_ = [0]*60
        #print(p_)
        #print(ds['state_ps'][col])
        p_[0] = ds['state_ps'][col]
        p_ = [(p[i+1] - p[i])/2 for i in range(59)]
        dp = [abs((p_[i+1] - p[i])) for i in range(58)]
        #print(dp)
        return dp


    
def plot_map(grid_info, grid_b, file_merged, file_input = "", var=['area'], lev_=59, reset=False):

    if file_input == "": file_input = file_merged

    if (file_input[-32]) != 'E':
            print("/!\ make sure to keep the original filename of the data file /!\ \n should be E3SM-MMF.mlX.XXXX-XX-XX-XXXXX.nc")
            return 

    if os.path.isfile(grid_b) == False or reset: create_grid_info(grid_b, grid_info)

    if file_input[-21] == 'i':
        if os.path.isfile(file_merged) == False or reset: create_visu_input(grid_b, file_input, file_merged)
    elif file_input[-21] == 'o':
        print("merging grid and data ..")
        if os.path.isfile(file_merged) == False or reset: create_visu_output(grid_b, file_input, file_merged)
    else:
        print("/!\ make sure to keep the original filename of the data file /!\ \n should be E3SM-MMF.mlX.XXXX-XX-XX-XXXXX.nc")
        return

    psy.rcParams['plotter.maps.xgrid'] = False
    psy.rcParams['plotter.maps.ygrid'] = False
    mpl.rcParams['figure.figsize'] = [14., 10.]


    ps_data = psyplot.open_dataset(file_merged)
    res = 'hr' if ps_data['area'].shape[0] >= 10000 else 'lr'

    

    fout = 'map/map_'+var[0]+'_'+file_input[-17:-3]+'_'+res+'.png' if len(ps_data[var[0]].shape) <=1  else 'map/map_'+var[0]+'_lev'+str(lev_)+'_'+file_input[-17:-3]+'_'+res+'.png' 

     
    # for i, vari in enumerate(ps_data.data_vars):
    #     if vari=='ptend_t' : ps_data[vari] = np.abs(ps_data[vari])

    print("saving map to ",fout)
    print("\n\n")
    if len(var)>1 :
        maps = ps_data.isel(lev =lev_).psy.plot.mapcombined(
        name=[var], load=True,
        arrowsize=1000,
        clabel='{desc}', stock_img=True, datagrid=dict(color='k', linewidth=0.2), projection='robin')
        maps.export(fout)

    else :
   
        maps = ps_data.isel(lev =lev_).psy.plot.mapplot(
        name=var[0], load=True, cmap='rainbow',
        clabel='{desc}', stock_img=True, datagrid=dict(color='k', linewidth=0.2), projection='robin')
        maps.export(fout)

    psy.close('all')

