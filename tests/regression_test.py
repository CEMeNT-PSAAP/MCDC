import os
import sys
import h5py
import numpy as np

sys.path.append('regression_data')

from context          import mcdc
from regression_model import ss_slab1, ss_slab2, ss_inf1, td_inf1


#==============================================================================
# Slab1
#==============================================================================

def test_regression_ss_slab1():
    # Setup
    ss_slab1.run()

    # Ans
    with h5py.File('output.h5', 'r') as f:
        x  = f['tally/spatial_grid'][:]
        dx = (x[1]-x[0])
        phi         = f['tally/flux/mean'][:]/dx
        phi_sd      = f['tally/flux/sdev'][:]/dx
        phi_face    = f['tally/flux-face/mean'][1:]
        phi_face_sd = f['tally/flux-face/sdev'][1:]
    os.remove('output.h5')

    # Sol
    with h5py.File('regression_data/ss_slab1.h5', 'r') as f:
        x  = f['tally/spatial_grid'][:]
        dx = (x[1]-x[0])
        phi_ref         = f['tally/flux/mean'][:]/dx
        phi_sd_ref      = f['tally/flux/sdev'][:]/dx
        phi_face_ref    = f['tally/flux-face/mean'][1:]
        phi_face_sd_ref = f['tally/flux-face/sdev'][1:]

    assert phi.all() == phi_ref.all()
    assert phi_sd.all() == phi_sd_ref.all()
    assert phi_face.all() == phi_face_ref.all()
    assert phi_face_sd.all() == phi_face_sd_ref.all()

def test_regression_slab1_vrt():
    # Setup
    ss_slab1.set_vrt(continuous_capture=True)
    ss_slab1.run()

    # Ans
    with h5py.File('output.h5', 'r') as f:
        x  = f['tally/spatial_grid'][:]
        dx = (x[1]-x[0])
        phi         = f['tally/flux/mean'][:]/dx
        phi_sd      = f['tally/flux/sdev'][:]/dx
        phi_face    = f['tally/flux-face/mean'][1:]
        phi_face_sd = f['tally/flux-face/sdev'][1:]
    os.remove('output.h5')

    # Sol
    with h5py.File('regression_data/ss_slab1_vrt.h5', 'r') as f:
        x  = f['tally/spatial_grid'][:]
        dx = (x[1]-x[0])
        phi_ref         = f['tally/flux/mean'][:]/dx
        phi_sd_ref      = f['tally/flux/sdev'][:]/dx
        phi_face_ref    = f['tally/flux-face/mean'][1:]
        phi_face_sd_ref = f['tally/flux-face/sdev'][1:]

    assert phi.all() == phi_ref.all()
    assert phi_sd.all() == phi_sd_ref.all()
    assert phi_face.all() == phi_face_ref.all()
    assert phi_face_sd.all() == phi_face_sd_ref.all()

#==============================================================================
# Slab2
#==============================================================================

def test_regression_ss_slab2():
    # Setup
    ss_slab2.run()

    # Ans
    with h5py.File('output.h5', 'r') as f:
        x  = f['tally/spatial_grid'][:]
        dx = (x[1]-x[0])
        phi         = f['tally/flux/mean'][:]/dx
        phi_sd      = f['tally/flux/sdev'][:]/dx
        phi_face    = f['tally/flux-face/mean'][:]
        phi_face_sd = f['tally/flux-face/sdev'][:]
        J         = f['tally/current/mean'][:,0]/dx
        J_sd      = f['tally/current/sdev'][:,0]/dx
        J_face    = f['tally/current-face/mean'][:,0]
        J_face_sd = f['tally/current-face/sdev'][:,0]
    os.remove('output.h5')

    # Sol
    with h5py.File('regression_data/ss_slab2.h5', 'r') as f:
        x  = f['tally/spatial_grid'][:]
        dx = (x[1]-x[0])
        phi_ref         = f['tally/flux/mean'][:]/dx
        phi_sd_ref      = f['tally/flux/sdev'][:]/dx
        phi_face_ref    = f['tally/flux-face/mean'][:]
        phi_face_sd_ref = f['tally/flux-face/sdev'][:]
        J_ref         = f['tally/current/mean'][:,0]/dx
        J_sd_ref      = f['tally/current/sdev'][:,0]/dx
        J_face_ref    = f['tally/current-face/mean'][:,0]
        J_face_sd_ref = f['tally/current-face/sdev'][:,0]

    assert phi.all() == phi_ref.all()
    assert phi_sd.all() == phi_sd_ref.all()
    assert phi_face.all() == phi_face_ref.all()
    assert phi_face_sd.all() == phi_face_sd_ref.all()
    assert J.all() == J_ref.all()
    assert J_sd.all() == J_sd_ref.all()
    assert J_face.all() == J_face_ref.all()
    assert J_face_sd.all() == J_face_sd_ref.all()

def test_regression_slab2_vrt():
    # Setup
    ss_slab2.set_vrt(continuous_capture=True)
    ss_slab2.run()

    # Ans
    with h5py.File('output.h5', 'r') as f:
        x  = f['tally/spatial_grid'][:]
        dx = (x[1]-x[0])
        phi         = f['tally/flux/mean'][:]/dx
        phi_sd      = f['tally/flux/sdev'][:]/dx
        phi_face    = f['tally/flux-face/mean'][:]
        phi_face_sd = f['tally/flux-face/sdev'][:]
        J         = f['tally/current/mean'][:,0]/dx
        J_sd      = f['tally/current/sdev'][:,0]/dx
        J_face    = f['tally/current-face/mean'][:,0]
        J_face_sd = f['tally/current-face/sdev'][:,0]
    os.remove('output.h5')

    # Sol
    with h5py.File('regression_data/ss_slab2_vrt.h5', 'r') as f:
        x  = f['tally/spatial_grid'][:]
        dx = (x[1]-x[0])
        phi_ref         = f['tally/flux/mean'][:]/dx
        phi_sd_ref      = f['tally/flux/sdev'][:]/dx
        phi_face_ref    = f['tally/flux-face/mean'][:]
        phi_face_sd_ref = f['tally/flux-face/sdev'][:]
        J_ref         = f['tally/current/mean'][:,0]/dx
        J_sd_ref      = f['tally/current/sdev'][:,0]/dx
        J_face_ref    = f['tally/current-face/mean'][:,0]
        J_face_sd_ref = f['tally/current-face/sdev'][:,0]

    assert phi.all() == phi_ref.all()
    assert phi_sd.all() == phi_sd_ref.all()
    assert phi_face.all() == phi_face_ref.all()
    assert phi_face_sd.all() == phi_face_sd_ref.all()
    assert J.all() == J_ref.all()
    assert J_sd.all() == J_sd_ref.all()
    assert J_face.all() == J_face_ref.all()
    assert J_face_sd.all() == J_face_sd_ref.all()

#==============================================================================
# Inf1
#==============================================================================

def test_regression_ss_inf1():
    # Setup
    ss_inf1.run()

    # Ans
    with np.load('regression_data/XS_inf1.npz') as data:
        E = data['E']        # eV
    E_mid = 0.5*(E[:-1] + E[1:])
    dE    = E[1:] - E[:-1]
    with h5py.File('output.h5', 'r') as f:
        phi    = f['tally/flux/mean'][:]/dE*E_mid
        phi_sd = f['tally/flux/sdev'][:]/dE*E_mid              
    os.remove('output.h5')

    # Sol
    with h5py.File('regression_data/ss_inf1.h5', 'r') as f:
        phi_ref    = f['tally/flux/mean'][:]/dE*E_mid
        phi_sd_ref = f['tally/flux/sdev'][:]/dE*E_mid              

    assert phi.all() == phi_ref.all()
    assert phi_sd.all() == phi_sd_ref.all()

def test_regression_ss_inf1_vrt_capture():
    # Setup
    ss_inf1.set_vrt(continuous_capture=True, wgt_roulette=0.25)
    ss_inf1.run()

    # Ans
    with np.load('regression_data/XS_inf1.npz') as data:
        E = data['E']        # eV
    E_mid = 0.5*(E[:-1] + E[1:])
    dE    = E[1:] - E[:-1]
    with h5py.File('output.h5', 'r') as f:
        phi    = f['tally/flux/mean'][:]/dE*E_mid
        phi_sd = f['tally/flux/sdev'][:]/dE*E_mid              
    os.remove('output.h5')

    # Sol
    with h5py.File('regression_data/ss_inf1_vrt_capture.h5', 'r') as f:
        phi_ref    = f['tally/flux/mean'][:]/dE*E_mid
        phi_sd_ref = f['tally/flux/sdev'][:]/dE*E_mid              

    assert phi.all() == phi_ref.all()
    assert phi_sd.all() == phi_sd_ref.all()

def test_regression_ss_inf1_vrt_fission():
    # Setup
    ss_inf1.set_vrt(continuous_capture=False,implicit_fission=True, wgt_roulette=0.25)
    ss_inf1.run()

    # Ans
    with np.load('regression_data/XS_inf1.npz') as data:
        E = data['E']        # eV
    E_mid = 0.5*(E[:-1] + E[1:])
    dE    = E[1:] - E[:-1]
    with h5py.File('output.h5', 'r') as f:
        phi    = f['tally/flux/mean'][:]/dE*E_mid
        phi_sd = f['tally/flux/sdev'][:]/dE*E_mid              
    os.remove('output.h5')

    # Sol
    with h5py.File('regression_data/ss_inf1_vrt_fission.h5', 'r') as f:
        phi_ref    = f['tally/flux/mean'][:]/dE*E_mid
        phi_sd_ref = f['tally/flux/sdev'][:]/dE*E_mid              

    assert phi.all() == phi_ref.all()
    assert phi_sd.all() == phi_sd_ref.all()

def test_regression_ss_inf1_vrt_capture_fission():
    # Setup
    ss_inf1.set_vrt(continuous_capture=True,implicit_fission=True,wgt_roulette=0.25)
    ss_inf1.run()

    # Ans
    with np.load('regression_data/XS_inf1.npz') as data:
        E = data['E']        # eV
    E_mid = 0.5*(E[:-1] + E[1:])
    dE    = E[1:] - E[:-1]
    with h5py.File('output.h5', 'r') as f:
        phi    = f['tally/flux/mean'][:]/dE*E_mid
        phi_sd = f['tally/flux/sdev'][:]/dE*E_mid              
    os.remove('output.h5')

    # Sol
    with h5py.File('regression_data/ss_inf1_vrt_capture_fission.h5', 'r') as f:
        phi_ref    = f['tally/flux/mean'][:]/dE*E_mid
        phi_sd_ref = f['tally/flux/sdev'][:]/dE*E_mid              

    assert phi.all() == phi_ref.all()
    assert phi_sd.all() == phi_sd_ref.all()
