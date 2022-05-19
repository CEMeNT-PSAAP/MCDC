import numpy as np
import h5py

def test():
    exec(open("input.py").read())

    with h5py.File('output.h5', 'r') as f:
        phi_x    = f['tally/flux-x/mean'][:]
        phi_x_sd = f['tally/flux-x/sdev'][:]
        k        = f['keff'][:]
    
    with h5py.File('output.ans', 'r') as f:
        phi_x_exp    = f['tally/flux-x/mean'][:]
        phi_x_sd_exp = f['tally/flux-x/sdev'][:]
        k_exp        = f['keff'][:]

    assert np.array_equal(phi_x,phi_x_exp)
    assert np.array_equal(phi_x_sd,phi_x_sd_exp)
    assert np.array_equal(k,k_exp)
