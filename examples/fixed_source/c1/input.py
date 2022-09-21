import math
import numpy as np
import mcdc

# =============================================================================
# Set model
# =============================================================================
v = 29.9792458
pi = math.acos(-1)

# Set materials
m = mcdc.material(capture=np.array([0.1]),
                  scatter=np.array([[0.9]]),
                  speed=np.array([v]))

#mcdc.universal_speed(v)

# Set surfaces
sx1 = mcdc.surface('plane-x', x=0.0, bc="vacuum")
sx2 = mcdc.surface('plane-x', x=1.0, bc="vacuum")
sy1 = mcdc.surface('plane-y', y=0.0, bc="vacuum")
sy2 = mcdc.surface('plane-y', y=1.0, bc="vacuum")

# Set cells
mcdc.cell([+sx1, -sx2, +sy1, -sy2], m)

# =============================================================================
# Set sources
# =============================================================================

#initial condition
mcdc.source(x=[0.0, 1.0],
            y=[0.0, 1.0],
            prob=pi*4E-2,
            isotropic=True)

#source
mcdc.source(time=[0.0, 1.0],
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            prob=v,
            isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================
mcdc.tally(scores=['flux','flux-x','flux-y','flux-t',
                   'n','n-x','n-y','n-t'], 
           x=np.linspace(0.0, 1.0, 11), 
           y=np.linspace(0.0, 1.0, 11),
           t=np.linspace(0.0, 1.0, 51))

# Setting
mcdc.setting(N_particle=1E7,
             time_boundary=1.0,
             bank_max=2E7)


mcdc.implicit_capture()
#mcdc.ww_source()
#mcdc.population_control()


#f = np.load("ww.npz")
#mcdc.weight_window(x=np.linspace(0.0, 1.0, 11),
#                   y=np.linspace(0.0, 1.0, 11),
#                   t=np.linspace(0.0, 1.0, 51),
#										window=f['phi'])
mcdc.run()