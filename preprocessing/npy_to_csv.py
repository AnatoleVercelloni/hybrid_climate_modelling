import numpy as np
import glob
import pandas as pd


path = '/gpfsscratch/rech/psl/upu87pm/test_data_from_kaggle/'


L = []
for i in range(50):
    print(np.load(path + 'test_id'+str(i).rjust(2,'0')+'.npy').shape)
    L.append(np.concatenate([np.load(path + 'test_id'+str(i).rjust(2,'0')+'.npy').reshape(12500, 1), np.load(path + 'test_'+str(i).rjust(2,'0')+'.npy')], axis = 1))
    print (L[i].shape)

pred = np.concatenate(L, axis = 0)
print(pred.shape)

df = pd.DataFrame(pred)
df.to_csv('/gpfsscratch/rech/psl/upu87pm/predictions/MLP/for_kaggle/sample_submission.csv')