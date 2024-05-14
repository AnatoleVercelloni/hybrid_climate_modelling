from tonetcdf import create_grid_info, create_visu_input, create_visu_output
import tensorflow as tf
import xarray as xr

vars_mli = ['state_t','state_q0001','state_ps','pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
vars_mlo = ['ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC','cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']

path = '/gpfswork/rech/psl/upu87pm/ClimSim'
mlo_scale = xr.open_dataset(path + '/preprocessing/normalizations/outputs/output_scale.nc')
mli_mean = xr.open_dataset(path + '/preprocessing/normalizations/inputs/input_mean.nc')
mli_min = xr.open_dataset(path + '/preprocessing/normalizations/inputs/input_min.nc')
mli_max = xr.open_dataset(path + '/preprocessing/normalizations/inputs/input_max.nc')


def setup_visu(grid, grid_b, file_input, file_imerged):

    print("creating low resolution grid boundaries..")
    create_grid_info(grid_b, grid)


    create_visu_input(grid_b, file_input, file_imerged)
    print(file_imerged+ "  created !")

    create_visu_output(grid_b, file_input.replace('.mli.','.mlo.'), file_imerged.replace('.mli.','.mlo.'))
    print(file_imerged.replace('.mli.','.mlo.')+ "  created !")



#function defined in climsim_utils.data_utils, I should be able to import it 


def load_nc_dir_with_generator(filelist:list):
    '''
        This function works as a dataloader when training the emulator with raw netCDF files.
        This can be used as a dataloader during training or it can be used to create entire datasets. !!
        When used as a dataloader for training, I/O can slow down training considerably.
        This function also normalizes the data.
        mli corresponds to input
        mlo corresponds to target
        '''
    def gen():
        for file in filelist:
            
            # read mli
            ds = xr.open_dataset(file, engine='netcdf4')
            ds = ds[vars_mli]                             
            
            # read mlo
            dso = xr.open_dataset(file.replace('.mli.','.mlo.'), engine='netcdf4') 
            
            # make mlo variales: ptend_t and ptend_q0001
            dso['ptend_t'] = (dso['state_t'] - ds['state_t'])/1200 # T tendency [K/s]
            dso['ptend_q0001'] = (dso['state_q0001'] - ds['state_q0001'])/1200 # Q tendency [kg/kg/s]
            dso = dso[vars_mlo]
            
            # normalizatoin, scaling
            ds = (ds-mli_mean)/(mli_max-mli_min)
            dso = dso*mlo_scale

            # stack
            #ds = ds.stack({'batch':{'sample','ncol'}})
            ds = ds.stack({'batch':{'ncol'}})
            ds = ds.to_stacked_array("mlvar", sample_dims=["batch"], name='mli')
            #dso = dso.stack({'batch':{'sample','ncol'}})
            dso = dso.stack({'batch':{'ncol'}})
            dso = dso.to_stacked_array("mlvar", sample_dims=["batch"], name='mlo')
            
            yield (ds.values, dso.values)

    return tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float64, tf.float64),
        output_shapes=((None,124),(None,128))
    )