#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 14:29:02 2022

@author: sampasmann
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

# =============================================================================
# import cross sections 
# =============================================================================
G = 12 # G may equal 12, 70, or 618
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path        = "./HDPE/"
abs_file_path   = os.path.join(script_dir, rel_path)
D = np.genfromtxt(abs_file_path+"D_{}G_HDPE.csv".format(G), delimiter=",")
SigmaA = np.genfromtxt(abs_file_path+"Siga_{}G_HDPE.csv".format(G), delimiter=",")
SigmaS = np.genfromtxt(abs_file_path+"Scat_{}G_HDPE.csv".format(G), delimiter=",")
# SigmaS = np.flip(SigmaS,1)
# SigmaS = np.flip(SigmaS)

# =============================================================================
# Calculate analytic solution
# =============================================================================
Nx = 5
source = np.ones((G,Nx))
SigmaT = SigmaS.sum(axis=0) + SigmaA
SigmaT = 1/(3*D)
sol = np.dot(np.linalg.inv(np.diag(SigmaT) - SigmaS), source)

# =============================================================================
# Grab data from MCDC Run
# =============================================================================

# hf = h5py.File('/Users/sampasmann/Documents/GitHub/MCDC/examples/fixed_source/inf_hdpe_iqmc/output.h5', 'r')
hf = h5py.File('C:/Users/Sam/Documents/Github/MCDC/examples/fixed_source/inf_hdpe_iqmc/output.h5', 'r')

G           = 12
flux        = hf['tally']['iqmc_flux'][:]
xspan       = hf['iqmc']['grid']['x'][:]
dx          = xspan[1] - xspan[0]
midpoints   = xspan[1:] - dx
Nx          = flux.shape[1]

# =============================================================================
# Plot 
# =============================================================================
plt.figure(dpi=300, figsize=(7,4))
for i in range(G):
    plt.plot(midpoints,flux[i,:])
    # plt.plot(midpoints, sol[i,:], '--')
plt.title('HDPE Multigroup Problem')
plt.ylabel(r'$\phi$')
plt.xlabel('x')

hf.close()