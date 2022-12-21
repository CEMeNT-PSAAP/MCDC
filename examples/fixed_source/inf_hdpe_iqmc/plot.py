#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 14:29:02 2022

@author: sampasmann
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt

# Load XS
with np.load('../td_inf_shem361/SHEM-361.npz') as data:
    SigmaT   = data['SigmaT']
    nuSigmaF = data['nuSigmaF']
    SigmaS   = data['SigmaS']
    E        = data['E']

G = len(SigmaT)
E_mid = 0.5*(E[1:]+E[:-1])
dE    = E[1:]-E[:-1]

hf = h5py.File('/Users/sampasmann/Documents/GitHub/MCDC/examples/fixed_source/inf_hdpe_iqmc/output.h5', 'r')

flux = hf['tally']['iqmc_flux'][:]
flux = flux/dE*E_mid
xspan = hf['iqmc']['grid']['x'][:]
dx = xspan[1] - xspan[0]
midpoints = xspan[1:] - dx
# Nx = flux.shape[1]


plt.step(E_mid,flux,'-b',label="MC",where='mid')
# plt.fill_between(E_mid,phi_avg-phi_sd,phi_avg+phi_sd,alpha=0.2,color='b',step='mid')
# plt.step(E_mid,phi_exact,'--r',label='analytical',where='mid')
plt.xscale('log')
plt.xlabel(r'$E$, eV')
plt.ylabel(r'$E\phi(E)$')
plt.grid()
plt.legend()
plt.show()
# plt.figure(dpi=300, figsize=(7,4))
# plt.plot(flux)
# plt.title('HDPE Multigroup Problem')
# plt.ylabel(r'$\phi$')
# plt.xlabel('x')

hf.close()