import numpy as np
import matplotlib.pyplot as plt
import h5py


E = np.array([1.440E-01, 1.860E-01, 7.660E-01, 1.001E+00])*1E6

# MC/DC Results
with h5py.File('output.h5', 'r') as f:
    score    = f['tally/total_crossing/mean'][:]*1E6
    score_sd = f['tally/total_crossing/sdev'][:]*1E6
rad    = score[:,0]
rad_sd = score_sd[:,0]
top    = score[:,1]
top_sd = score_sd[:,1]

# Ref.
ref_rad    = np.array([6.72E1, 6.72E2, 6.72E-1, 6.72])
ref_rad_sd = ref_rad*np.array([10.46E-2, 8.02E-2, 4.7E-2, 2.46E-2])
ref_top    = np.array([6.63E1, 1.21E3, 2.12, 8.23])
ref_top_sd = ref_top*np.array([12.32E-2, 8.71E-2, 3.19E-2, 1.84E-2])

# Plot
plt.figure(figsize=(4,3))
x = np.arange(len(E))
plt.plot(x,top,'-b',label="Axial-Top")
plt.fill_between(x,top-top_sd,top+top_sd,alpha=0.2,color='b')
plt.plot(x,rad,'-r',label="Radial")
plt.fill_between(x,rad-rad_sd,rad+rad_sd,alpha=0.2,color='r')
plt.plot(x,ref_top,'--b',label="Axial-Top, Point (Ref.)")
plt.fill_between(x,ref_top-ref_top_sd,ref_top+ref_top_sd,alpha=0.2,color='b')
plt.plot(x,ref_rad,'--r',label="Radial, Ring (Ref.)")
plt.fill_between(x,ref_rad-ref_rad_sd,ref_rad+ref_rad_sd,alpha=0.2,color='r')
plt.yscale('log')
plt.xlabel('Energy group')
plt.ylabel('Normalized Counts')
plt.grid()
plt.legend()
plt.show()
