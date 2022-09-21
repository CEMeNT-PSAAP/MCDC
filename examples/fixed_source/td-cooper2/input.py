import math
import numpy as np
import mcdc

pi = math.acos(-1)
# =============================================================================
# Set model
# =============================================================================
v=1.383

# Set materials
m         = mcdc.material(capture=np.array([0.2]),
                          scatter=np.array([[0.8]]),
                          speed=np.array([[v]]))
m_barrier = mcdc.material(capture=np.array([1.0]),
                          scatter=np.array([[4.0]]),
                          speed=np.array([[v]]))

#mcdc.universal_speed(v) #broken

# Set surfaces
sx1 = mcdc.surface('plane-x', x=0.0,  bc="reflective")
sx2 = mcdc.surface('plane-x', x=10.0)
sx3 = mcdc.surface('plane-x', x=12.0)
sx4 = mcdc.surface('plane-x', x=20.0, bc="vacuum")
sy1 = mcdc.surface('plane-y', y=0.0,  bc="reflective")
sy2 = mcdc.surface('plane-y', y=10.0)
sy3 = mcdc.surface('plane-y', y=20.0, bc="vacuum")

# Set cells
mcdc.cell([+sx1, -sx2, +sy1, -sy2], m)
mcdc.cell([+sx3, -sx4, +sy1, -sy2], m)
mcdc.cell([+sx1, -sx4, +sy2, -sy3], m)
mcdc.cell([+sx2, -sx3, +sy1, -sy2], m_barrier)

# =============================================================================
# Set source
# =============================================================================

#initial condition
mcdc.source(x=[0.0, 20.0], y=[0.0, 20.0], prob=20.0*20.0/1.0E1, weight=pi*4.0E-10*1.0E1, isotropic=True)

#source
#mcdc.source(time=[0.0, 2.0], x=[0.0, 5.0], y=[0.0, 5.0], prob=2.0*v*5*5, isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================
t=np.linspace(0.0,2.0,21)
#t[0]=t[0]+1E-6
#t[-1]=t[-1]-1E-6

mcdc.tally(scores=['flux','flux-x','flux-y','flux-t',
                   'n','n-x','n-y','n-t'], 
           x=np.linspace(0.0, 20.0, 41), 
           y=np.linspace(0.0, 20.0, 41),
           t=t)

# Setting
mcdc.setting(N_particle=2E5,
             time_boundary=2.0,
             bank_max=2E7,
             rng_seed=12345)

# Technique
mcdc.implicit_capture()
#mcdc.population_control()

f = np.load("ww.npz")
#with np.printoptions(threshold=np.inf):
#	print(f['phi'][0][0][11])
mcdc.weight_window(x=np.linspace(0.0, 20.0, 41),
                   y=np.linspace(0.0, 20.0, 41),
                   t=np.linspace(0.0, 2.0, 21),
                   f=1.0E2,
                   window=f['phi'])

mcdc.run()
