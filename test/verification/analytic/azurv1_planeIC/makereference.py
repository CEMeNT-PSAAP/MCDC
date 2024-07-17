import numpy as np
import h5py
import scipy

# ======================================================================================================
# This script loads benchmark solutions at the same x values as output of MC/DC, but instantaneous in t (from 0 to 20, with step .002)
# MC/DC (as implemented for this problem) outputs flux across 1 second intervals;
# This script integrates values that are instantaneous in t into flux across the same 1 second intervals
# ======================================================================================================

x = np.linspace(-20.5, 20.5, 202)
dx = x[1:] - x[:-1]
x = x[1:] - dx / 2
t = np.linspace(0.5, 19.5, 20)
phi = np.empty((100001, 201))
tavg = np.linspace(0, 20, 100001)
with h5py.File("../benchmarks.hdf5", "r") as f:
    for i in range(100001):
        key = "plane_IC/t = " + str(tavg[i])
        phi[i, :] = f[key][1, :]
phiavg = np.empty((20, 201))
# The following loops turn the benchmark solutions (flux values instantaneously in x and t) to values similar to the MC/DC output (flux across t)
for i in range(20):  # t values in MC/DC
    for j in range(201):  # x values (same in both)
        # benchmark time step is .002, this performs integration across 1 second
        if i == 0:
            # This is a special case: because the initial condition is a delta function, the integrand does not behave well around x = 0
            # This is mitigated by beginning the integration later for that one x "bucket" at that one time "bucket"
            if np.fabs(x[j]) - 0.25 < 0:
                phiavg[i, j] = scipy.integrate.simpson(
                    y=phi[(5000 * i + 200) : (5000 * i + 5000), j],
                    x=tavg[(5000 * i + 200) : (5000 + 5000 * i)],
                )
            else:
                phiavg[i, j] = scipy.integrate.simpson(
                    y=phi[(5000 * i) : (5000 * i + 5000), j],
                    x=tavg[(5000 * i) : (5000 + 5000 * i)],
                )
        else:
            phiavg[i, j] = scipy.integrate.simpson(
                y=phi[(5000 * i) : (5000 * i + 5000), j],
                x=tavg[(5000 * i) : (5000 + 5000 * i)],
            )
np.savez("reference.npz", x=x, t=t, phi=phiavg)
