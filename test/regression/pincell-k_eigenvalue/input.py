import mcdc
import numpy as np
import os

os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/"

# Create MC/DC simulation
simulation = mcdc.Simulation("Pincell k-eigenvalue")

# Material
fuel = mcdc.Material(
    nuclide_composition={
        "U235": 0.0001654509603995036,
        "U238": 0.022801089905717036,
        "O16": 0.04593308173223308,
    }
)
moderator = mcdc.Material(
    nuclide_composition={
        "H1": 0.05129627050184732,
        "O16": 0.024622209840886707,
        "B10": 4.103701640147785e-05,
    }
)

# Geometry
cylinder = mcdc.Surface.CylinderZ(radius=0.45720)
pitch = 1.25984
x0 = mcdc.Surface.PlaneX(x=-pitch / 2, boundary_condition="reflective")
x1 = mcdc.Surface.PlaneX(x=pitch / 2, boundary_condition="reflective")
y0 = mcdc.Surface.PlaneY(y=-pitch / 2, boundary_condition="reflective")
y1 = mcdc.Surface.PlaneY(y=pitch / 2, boundary_condition="reflective")
#
fuel_cell = mcdc.Cell(-cylinder, fill=fuel)
moderator_cell = mcdc.Cell(+x0 & -x1 & +y0 & -y1 & +cylinder, fill=moderator)
simulation.set_model([fuel_cell, moderator_cell])

# Source
source = mcdc.Source(position=[0.0, 0.0, 0.0], isotropic=True, energy=14.1e6)
simulation.set_sources([source])

# Setting
simulation.settings.N_particle = 30
simulation.settings.active_bank_buffer = 1000
simulation.settings.census_bank_buffer_ratio = 3.0
simulation.settings.source_bank_buffer_ratio = 3.0
simulation.settings.set_eigenmode(N_inactive=1, N_active=2)

# Tally
e_min, e_max = 1e-5, 20.0e6
groups = 500
energies = np.logspace(np.log10(e_min), np.log10(e_max), groups + 1)

tally = mcdc.Tally(scores=["flux"], energy=energies)
simulation.set_tallies([tally])

simulation.run()
