import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.integrate import quad


# =============================================================================
# Plot results
# =============================================================================

# Load grids
with h5py.File('output.h5', 'r') as f:
    x = f['tally1/spatial_grid'][:]
dx    = (x[1]-x[0])
x_mid = 0.5*(x[:-1]+x[1:])

# Results
with h5py.File('output.h5', 'r') as f:
    phi         = f['tally1/flux/mean'][:]/dx/2
    phi_sd      = f['tally1/flux/sdev'][:]/dx/2
    phi_face    = f['tally1/flux-face/mean'][:]/2
    phi_face_sd = f['tally1/flux-face/sdev'][:]/2
    J         = f['tally1/current/mean'][:,0]/dx/2
    J_sd      = f['tally1/current/sdev'][:,0]/dx/2
    J_face    = f['tally1/current-face/mean'][:,0]/2
    J_face_sd = f['tally1/current-face/sdev'][:,0]/2

    leak_left     = f['tally2/partial_crossing/mean'][0,0]/2
    leak_left_sd  = f['tally2/partial_crossing/sdev'][0,0]/2
    leak_right    = f['tally2/partial_crossing/mean'][1,1]/2
    leak_right_sd = f['tally2/partial_crossing/sdev'][1,1]/2

print("leak left:",leak_left,"+/-",leak_left_sd)
print("leak right:",leak_right,"+/-",leak_right_sd)

# Plot
plt.plot(x_mid,phi,'-b',label="MC")
plt.fill_between(x_mid,phi-phi_sd,phi+phi_sd,alpha=0.2,color='b')
plt.xlabel(r'$x$, cm')
plt.ylabel('Flux')
plt.grid()
plt.legend()
plt.title(r'$\bar{\phi}_i$')
plt.show()

plt.plot(x[:],phi_face,'-b',label="MC")
plt.fill_between(x[:],phi_face-phi_face_sd,phi_face+phi_face_sd,alpha=0.2,color='b')
plt.xlabel(r'$x$, cm')
plt.ylabel('Flux')
plt.grid()
plt.legend()
plt.title(r'$\phi(x)$')
plt.show()

# Solution
plt.plot(x_mid,J,'-b',label="MC")
plt.fill_between(x_mid,J-J_sd,J+J_sd,alpha=0.2,color='b')
plt.xlabel(r'$x$, cm')
plt.ylabel('Current')
plt.grid()
plt.legend()
plt.title(r'$\bar{J}_i$')
plt.show()

plt.plot(x[:],J_face,'-b',label="MC")
plt.fill_between(x[:],J_face-J_face_sd,J_face+J_face_sd,alpha=0.2,color='b')
plt.xlabel(r'$x$, cm')
plt.ylabel('Current')
plt.grid()
plt.legend()
plt.title(r'$J(x)$')
plt.show()
