import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.integrate import quad


# =============================================================================
# Reference solution
# =============================================================================

# Load grids
with h5py.File('output_1000.h5', 'r') as f:
    x = f['tally/grid/x'][:]
dx    = (x[1]-x[0])
x_mid = 0.5*(x[:-1]+x[1:])

# Parameters
SigmaT3 = 1.0
SigmaT1 = 1.5
SigmaT2 = 2.0
q1 = 1.0/6.0/2
q2 = 1.0/6.0/2
q3 = 1.0/6.0/2
x1 = 2.0
x2 = 4.0
x3 = 6.0
tau1 = SigmaT1*x1
tau2 = SigmaT2*(x2-x1)
tau3 = SigmaT3*(x3-x2)

# Angular flux
def psi1_plus(mu,x):
    return q1/SigmaT1*(1.0 - np.exp(-SigmaT1*x/mu))
def psi2_plus(mu,x):
    return (psi1_plus(mu,x1) - q2/SigmaT2)*np.exp(-SigmaT2*(x-x1)/mu) + q2/SigmaT2
def psi3_plus(mu,x):
    return (psi2_plus(mu,x2) - q3/SigmaT3)*np.exp(-SigmaT3*(x-x2)/mu) + q3/SigmaT3
def psi1_minus(mu,x):
    return (psi2_minus(mu,x1) - q1/SigmaT1)*np.exp(-SigmaT1*(x1-x)/np.abs(mu)) + q1/SigmaT1
def psi2_minus(mu,x):
    return (psi3_minus(mu,x2) - q2/SigmaT2)*np.exp(-SigmaT2*(x2-x)/np.abs(mu)) + q2/SigmaT2
def psi3_minus(mu,x):
    return q3/SigmaT3*(1.0 - np.exp(-SigmaT3*(x3-x)/np.abs(mu)))

# Flux
def phi1(x):
    return (quad(psi1_plus, 0, 1, args=(x))[0] + quad(psi1_minus, -1, 0, args=(x))[0])
def phi2(x):
    return (quad(psi2_plus, 0, 1, args=(x))[0] + quad(psi2_minus, -1, 0, args=(x))[0])
def phi3(x):
    return (quad(psi3_plus, 0, 1, args=(x))[0] + quad(psi3_minus, -1, 0, args=(x))[0])

# Integrands for current
def mu_psi1_plus(mu,x):
    return mu*psi1_plus(mu,x)
def mu_psi2_plus(mu,x):
    return mu*psi2_plus(mu,x)
def mu_psi3_plus(mu,x):
    return mu*psi3_plus(mu,x)
def mu_psi1_minus(mu,x):
    return mu*psi1_minus(mu,x)
def mu_psi2_minus(mu,x):
    return mu*psi2_minus(mu,x)
def mu_psi3_minus(mu,x):
    return mu*psi3_minus(mu,x)

# Current
def J1(x):
    return quad(mu_psi1_plus, 0, 1, args=(x))[0] + quad(mu_psi1_minus, -1, 0, args=(x))[0]
def J2(x):
    return quad(mu_psi2_plus, 0, 1, args=(x))[0] + quad(mu_psi2_minus, -1, 0, args=(x))[0]
def J3(x):
    return quad(mu_psi3_plus, 0, 1, args=(x))[0] + quad(mu_psi3_minus, -1, 0, args=(x))[0]

phi_ref      = np.zeros(60)
J_ref        = np.zeros(60)
phi_face_ref = np.zeros(61)
J_face_ref   = np.zeros(61)

for i in range(20):
    phi_ref[i]      = quad(phi1,x[i],x[i+1])[0]/dx
    J_ref[i]        = quad(J1,x[i],x[i+1])[0]/dx
    phi_face_ref[i] = phi1(x[i])
    J_face_ref[i]   = J1(x[i])
for i in range(20,40):
    phi_ref[i]      = quad(phi2,x[i],x[i+1])[0]/dx
    J_ref[i]        = quad(J2,x[i],x[i+1])[0]/dx
    phi_face_ref[i] = phi2(x[i])
    J_face_ref[i]   = J2(x[i])
for i in range(40,60):
    phi_ref[i]      = quad(phi3,x[i],x[i+1])[0]/dx
    J_ref[i]        = quad(J3,x[i],x[i+1])[0]/dx
    phi_face_ref[i] = phi3(x[i])
    J_face_ref[i]   = J3(x[i])
phi_face_ref[60] = phi3(x[60])
J_face_ref[60]   = J3(x[60])

# =============================================================================
# Plot results
# =============================================================================

error = []
N_hist_list = np.logspace(3, 10, 15)

for N_hist in N_hist_list:
    # Get results
    with h5py.File('output_%i.h5'%int(N_hist), 'r') as f:
        phi      = f['tally/flux/mean'][:]/dx
        phi_sd   = f['tally/flux/sdev'][:]/dx
        phi_x    = f['tally/flux-x/mean'][:]
        phi_x_sd = f['tally/flux-x/sdev'][:]
        J        = f['tally/current/mean'][:,0]/dx
        J_sd     = f['tally/current/sdev'][:,0]/dx
        J_x      = f['tally/current-x/mean'][:,0]
        J_x_sd   = f['tally/current-x/sdev'][:,0]
    
    error.append(np.linalg.norm((phi - phi_ref)/phi_ref))

line = 1.0/np.sqrt(N_hist_list)
line *= error[7]/line[7]
plt.plot(N_hist_list, error, 'bo', fillstyle='none')
plt.plot(N_hist_list, line, 'r--', label=r'$N^{-0.5}$')
plt.xscale('log')
plt.yscale('log')
plt.title(r'$\bar{\phi}_i$')
plt.ylabel('Flux 2-norm error')
plt.xlabel('# of histories')
plt.legend()
plt.grid()
plt.show()
