import numpy as np
import h5py

def test():
    exec(open("input.py").read())

    with h5py.File('output.h5', 'r') as f:
        phi      = f['tally/flux/mean'][:]
        phi_sd   = f['tally/flux/sdev'][:]
    
    with h5py.File('output.ans', 'r') as f:
        phi_exp      = f['tally/flux/mean'][:]
        phi_sd_exp   = f['tally/flux/sdev'][:]

    assert np.array_equal(phi,phi_exp)
    assert np.array_equal(phi_sd,phi_sd_exp)
