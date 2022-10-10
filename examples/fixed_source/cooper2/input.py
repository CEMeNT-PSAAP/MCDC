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

#source
mcdc.source(x=[0.0, 5.0], y=[0.0, 5.0], prob=v*5*5, isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

mcdc.tally(scores=['flux','n','current'],
           x=np.linspace(0.0, 20.0, 41), 
           y=np.linspace(0.0, 20.0, 41))

# Setting
mcdc.setting(N_particle=2E6,
             active_bank_buff=2.147E7, #max c int limit
             progress_bar=False,
             rng_seed=12345)

# Technique
mcdc.implicit_capture()

f = np.load("ww.npz")
mcdc.weight_window(x=np.linspace(0.0, 20.0, 41),
                   y=np.linspace(0.0, 20.0, 41),
                   f=1.0,
                   window=f['phi'],
                   Bx=f['Bx'],
                   By=f['By'],
                   Bz=f['Bz'])

mcdc.run()
