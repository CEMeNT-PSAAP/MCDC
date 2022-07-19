import numpy as np
import h5py
import os

def run():
    for mode in ['python', 'numba']:
        os.system("python input.py --mode "+mode)

        with h5py.File('output.ans', 'r') as f:
            group = f['tally/']
            names = [x[0] for x in list(group.items())]
        
        for name in names:
            if name == 'grid':
                continue
            with h5py.File('output.h5', 'r') as f:
                val    = f['tally/'+name+'/mean'][:]
                val_sd = f['tally/'+name+'/sdev'][:]
            with h5py.File('output.ans', 'r') as f:
                val_exp    = f['tally/'+name+'/mean'][:]
                val_sd_exp = f['tally/'+name+'/sdev'][:]
       
            assert np.array_equal(val,val_exp)
            assert np.array_equal(val_sd,val_sd_exp)
