import numpy as np
import csv
import pandas as pd


n_sample = 625000


v2_outputs = ['ptend_t',
              'ptend_q0001',
              'ptend_q0002',
              'ptend_q0003',
              'ptend_u',
              'ptend_v',
              'cam_out_NETSW',
              'cam_out_FLWDS',
              'cam_out_PRECSC',
              'cam_out_PRECC',
              'cam_out_SOLS',
              'cam_out_SOLL',
              'cam_out_SOLSD',
              'cam_out_SOLLD']




vertically_resolved = ['state_t', 
                       'state_q0001', 
                       'state_q0002', 
                       'state_q0003', 
                       'state_u', 
                       'state_v', 
                       'pbuf_ozone', 
                       'pbuf_CH4', 
                       'pbuf_N2O', 
                       'ptend_t', 
                       'ptend_q0001', 
                       'ptend_q0002', 
                       'ptend_q0003', 
                       'ptend_u', 
                       'ptend_v']



def csv_to_npy(normalize = False):

    path = '/gpfsscratch/rech/psl/upu87pm/'
    n_npy = 50
    i = 0

    for chunk in pd.read_csv(path + 'test.csv', chunksize=n_sample//n_npy):
        print(i)
        data_array = np.array(chunk)
        data_array = data_array[:,1:]
        data_array = np.array(data_array, dtype = float)
        
        print(data_array.shape)
        np.save(path + 'test_data_from_kaggle/test_id'+str(i).rjust(2,'0')+'.npy', data_array[:,1])

        np.save(path + 'test_data_from_kaggle/test_'+str(i).rjust(2,'0')+'.npy', data_array[:,1:])
        i = i + 1


def npy_to_csv():

    

    output_col_names = []
    for var in v2_outputs:
        if var in vertically_resolved:
            for i in range(60):
                output_col_names.append(var + '_' + str(i))
        else:
            output_col_names.append(var)


    path = '/gpfsscratch/rech/psl/upu87pm/test_data_from_kaggle/'
    prediction_path = '/gpfsscratch/rech/psl/upu87pm/predictions/MLP/for_kaggle/'

    colnames=['sample_id'] + output_col_names
    print("there is ", len(colnames), "colums (counting sample id)")
    L = []
    for i in range(50):
        print(np.load(path + 'test_id'+str(i).rjust(2,'0')+'.npy').shape)
        L.append(np.concatenate([np.load(path + 'test_id'+str(i).rjust(2,'0')+'.npy').reshape(12500, 1), np.load(prediction_path + 'ClimSim_model_'+str(i).rjust(2,'0')+'.npy')], axis = 1))
        print (L[i].shape)

    pred = np.concatenate(L, axis = 0)
    print(pred.shape)

    df = pd.DataFrame(pred)
    # df.colums = colnames
    df.to_csv('/gpfsscratch/rech/psl/upu87pm/predictions/MLP/for_kaggle/sample_submission.csv', header = colnames)


npy_to_csv()