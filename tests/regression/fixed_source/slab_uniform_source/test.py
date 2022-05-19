import numpy as np
import h5py

def test():
    exec(open("input.py").read())

    with h5py.File('output.h5', 'r') as f:
        phi      = f['tally/flux/mean'][:]
        phi_sd   = f['tally/flux/sdev'][:]
        phi_x    = f['tally/flux-x/mean'][:]
        phi_x_sd = f['tally/flux-x/sdev'][:]
        J        = f['tally/current/mean'][:]
        J_sd     = f['tally/current/sdev'][:]
        J_x      = f['tally/current-x/mean'][:]
        J_x_sd   = f['tally/current-x/sdev'][:]
    
    with h5py.File('output.ans', 'r') as f:
        phi_exp      = f['tally/flux/mean'][:]
        phi_sd_exp   = f['tally/flux/sdev'][:]
        phi_x_exp    = f['tally/flux-x/mean'][:]
        phi_x_sd_exp = f['tally/flux-x/sdev'][:]
        J_exp        = f['tally/current/mean'][:]
        J_sd_exp     = f['tally/current/sdev'][:]
        J_x_exp      = f['tally/current-x/mean'][:]
        J_x_sd_exp   = f['tally/current-x/sdev'][:]

    assert np.array_equal(phi,phi_exp)
    assert np.array_equal(phi_sd,phi_sd_exp)
    assert np.array_equal(phi_x,phi_x_exp)
    assert np.array_equal(phi_x_sd,phi_x_sd_exp)
    assert np.array_equal(J,J_exp)
    assert np.array_equal(J_sd,J_sd_exp)
    assert np.array_equal(J_x,J_x_exp)
    assert np.array_equal(J_x_sd,J_x_sd_exp)
