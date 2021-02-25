import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import quad


# =============================================================================
# Reference solution
# =============================================================================

c = 0.9
i = complex(0,1)

def integrand(u,eta,t):
    q   = (1+eta)/(1-eta)
    xi = (np.log(q)+i*u)/(eta+i*np.tan(u/2))
    return 1.0/(np.cos(u/2))**2*(xi**2*np.e**(c*t/2*(1-eta**2)*xi)).real
    
def phi(x,t):
    eta = x/t
    if abs(eta) >= 1.0:
        return 0.0
    else:
        integral = quad(integrand,0.0,np.pi,args=(eta,t))[0]
        return np.e**-t/2/t*(1+c*t/4/np.pi*(1-eta**2)*integral)

def phi_t(t,x):
    eta = x/t
    if abs(eta) >= 1.0:
        return 0.0
    else:
        integral = quad(integrand,0.0,np.pi,args=(eta,t))[0]
        return np.e**-t/2/t*(1+c*t/4/np.pi*(1-eta**2)*integral)

def phiX(x,t0,t1):
    return quad(phi_t,t0,t1,args=(x))[0]

x = np.linspace(-10.5,10.5,100)
x_mid = 0.5*(x[1:]+x[:-1])
t = [1E-10,1.0,2.0,10.0]
J = len(x_mid)
K = len(t)-1
phi_avg  = np.zeros([K,J])
phi_edge = np.zeros([K,J])
phi_face = np.zeros([K,J+1])

for k in range(K):
    for j in range(J):
        x0 = x[j]
        x1 = x[j+1]
        dx = x1-x0
        t0 = t[k]
        t1 = t[k+1]
        dt = t1-t0
        phi_edge[k,j] = quad(phi,x0,x1,args=(t1))[0]/dx
        phi_avg[k,j]  = quad(phiX,x0,x1,args=(t0,t1))[0]/dx/dt
    plt.plot(phi_avg[k])
    plt.show()
    assert()

for j in range(J+1):
    for k in range(K):
        t0 = t[k]
        t1 = t[k+1]
        dt = t1-t0
        phi_face[k,j] = quad(phi_t,t0,t1,args=(x[j]))[0]/dt
        
phi_edge = np.nan_to_num(phi_edge)
phi_face = np.nan_to_num(phi_face)
phi_avg = np.nan_to_num(phi_avg)
t = [0.0,1.0,2.0,10.0]
np.savez('azurv1_pl.npz',x=x,t=t,phi_edge=phi_edge,phi_face=phi_face,phi=phi_avg)