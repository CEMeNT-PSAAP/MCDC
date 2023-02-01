#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 14:29:02 2022

@author: sampasmann
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt

hf = h5py.File('/Users/sampasmann/Documents/GitHub/MCDC/examples/fixed_source/slab_reed_iqmc/output.h5', 'r')
# hf = h5py.File('C:/Users/Sam/Documents/Github/MCDC/examples/fixed_source/slab_reed_iqmc/output.h5', 'r')

flux = hf['tally']['iqmc_flux'][:]
xspan = hf['iqmc']['grid']['x'][:]
dx = xspan[1] - xspan[0]
midpoints = xspan[1:] - dx

plt.figure(dpi=300, figsize=(7,4))
plt.plot(midpoints, flux)
plt.title('Reeds Problem')
plt.ylabel(r'$\phi$')
plt.xlabel('x')

hf.close()