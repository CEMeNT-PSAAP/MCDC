import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

from reference import reference

output = sys.argv[1]

# Load results
output = sys.argv[1]
with np.load('CASMO-70.npz') as data:
    E     = data['E']
    G     = data['G']
    E_mid = 0.5*(E[1:]+E[:-1])
    dE    = E[1:]-E[:-1]
with h5py.File(output, 'r') as f:
    phi     = f['tally/flux/mean'][:]/dE*E_mid
    phi_sd  = f['tally/flux/sdev'][:]/dE*E_mid

# Reference solution
phi_ref = reference()/dE*E_mid

# Flux
plt.step(E_mid,phi,'-b',label="MC",where='mid')
plt.fill_between(E_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b',step='mid')
plt.step(E_mid,phi_ref,'--r',label='analytical',where='mid')
plt.xscale('log')
plt.xlabel(r'$E$, eV')
plt.ylabel(r'$E\phi(E)$')
plt.grid()
plt.legend()
plt.show()
