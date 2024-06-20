import numpy as np
import csv
import pandas as pd

path = '/gpfsscratch/rech/psl/upu87pm/'

n_sample = 625000
n_proc = 50



i = 0
for chunk in pd.read_csv(path + 'test.csv', chunksize=n_sample//n_proc):
    print(i)
    data_array = np.array(chunk)
    data_array = data_array[:,1:]
    data_array = np.array(data_array, dtype = float)
    
    print(data_array.shape)
    np.save(path + 'test_data_from_kaggle/test_id'+str(i).rjust(2,'0')+'.npy', data_array[:,1])

    np.save(path + 'test_data_from_kaggle/test_'+str(i).rjust(2,'0')+'.npy', data_array[:,1:])
    i = i + 1
