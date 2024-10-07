import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================
# Three slab layers with different materials
# Based on William H. Reed, NSE (1971), 46:2, 309-314, DOI: 10.13182/NSE46-309

# Set materials
m1 = mcdc.material(capture=np.array([50.0]))
m2 = mcdc.material(capture=np.array([5.0]))
m3 = mcdc.material(capture=np.array([0.0]))  # Vacuum
m4 = mcdc.material(capture=np.array([0.1]), scatter=np.array([[0.9]]))

# Set surfaces
s1 = mcdc.surface("plane-x", x=-8.0, bc="vacuum")
s2 = mcdc.surface("plane-x", x=-5.0)
s3 = mcdc.surface("plane-x", x=-3.0)
s4 = mcdc.surface("plane-x", x=-2.0)
s5 = mcdc.surface("plane-x", x=2.0)
s6 = mcdc.surface("plane-x", x=3.0)
s7 = mcdc.surface("plane-x", x=5.0)
s8 = mcdc.surface("plane-x", x=8.0, bc="vacuum")

# Set cells
mcdc.cell(+s1 & -s2, m4)
mcdc.cell(+s2 & -s3, m3)
mcdc.cell(+s3 & -s4, m2)
mcdc.cell(+s4 & -s5, m1)
mcdc.cell(+s5 & -s6, m2)
mcdc.cell(+s6 & -s7, m3)
mcdc.cell(+s7 & -s8, m4)


# =============================================================================
# iQMC Parameters
# =============================================================================
N = 1000
Nx = 64
maxit = 20
tol = 1e-1
x = np.linspace(-8, 8, num=Nx + 1)
solver = "gmres"


def reeds_source(Nx, LB=-8.0, RB=8.0):
    source = np.empty(Nx)
    dx = (RB - LB) / Nx
    xspan = np.linspace(LB + dx / 2, RB - dx / 2, num=Nx)
    count = 0
    for x in xspan:
        if x < -6:
            source[count] = 0.0
        elif (-6.0 < x) and (x < -5.0):
            source[count] = 1.0
        elif -5.0 < x < -3.0:  # vacuum region 1
            source[count] = 0.0
        elif -3.0 < x < -2.0:
            source[count] = 0.0
        elif -2.0 < x < 2.0:
            source[count] = 50.0
        elif 2.0 < x < 3.0:
            source[count] = 0.0
        elif 3.0 < x < 5.0:  # vacuum region 2
            source[count] = 0.0
        elif 5.0 < x < 6.0:
            source[count] = 1.0
        elif 6.0 < x:
            source[count] = 0.0
        count += 1

    return source


fixed_source = reeds_source(Nx)
phi0 = np.ones((Nx))

mcdc.iQMC(
    x=x,
    fixed_source=fixed_source,
    phi0=phi0,
    maxit=maxit,
    tol=tol,
    fixed_source_solver=solver,
    scores=["source-x"],
)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Setting
mcdc.setting(N_particle=N)

# Run
mcdc.run()
