import numpy as np
import os
import math
import mcdc
from datetime import datetime

# Set the XS library directory
os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/"

# Create MC/DC simulation
simulation = mcdc.Simulation("Lockwood")

# =============================================================================
# Set problem parameters
# =============================================================================
# Energy and Angle Parameters
MATERIAL_SYMBOL = "Al"
ENERGY = 1e4  # eV
CSDA_RANGE = 0.569  # g/cm2
ANGLE = 0.0

# MCDC Simulation Parameters
N_PARTICLES = 10
z0 = 0.0  # Starting source position

# Material Properties
RHO_G_CM3 = 2.70  # g/cm3
ATOMIC_WEIGHT_G_MOL = 26.7497084  # g/mol
AREAL_DENSITY_G_CM2 = 5.05e-3  # g/cm2

# Standard Calculations
dz = AREAL_DENSITY_G_CM2 / RHO_G_CM3
AVAGADRO_NUMBER = 6.02214076e23  # atoms/mol
MAT_DENSITY_ATOMS_PER_BARN_CM = (
    AVAGADRO_NUMBER / ATOMIC_WEIGHT_G_MOL * RHO_G_CM3 / 1e24
) * 1e-2  # atoms/barn-cm
TINY = 1e-30
L = CSDA_RANGE / RHO_G_CM3  # cm
N_LAYERS = 1
THETA = math.radians(ANGLE)

# Output variables for naming
np_name = len(str(N_PARTICLES)) - 1
e_name = f"{ENERGY:.2g}"

# =============================================================================
# Set materials
# =============================================================================
mat = mcdc.Material(
    element_composition={MATERIAL_SYMBOL: MAT_DENSITY_ATOMS_PER_BARN_CM}
)

# =============================================================================
# Set geometry (surfaces and cells)
# =============================================================================
# Z-direction surfaces for layers

s1 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")

s2 = mcdc.Surface.PlaneZ(z=L, boundary_condition="vacuum")

cell = mcdc.Cell(region=+s1 & -s2, fill=mat)
simulation.set_model([cell])

# =============================================================================
# Set source
# =============================================================================
# Parallel beam of 1 MeV electrons entering at z=0

source = mcdc.Source(
    z=[z0 + TINY, z0 + TINY],
    particle_type="electron",
    energy=np.array([[ENERGY - 1, ENERGY + 1], [0.5, 0.5]]),
    direction=[math.sin(THETA), 0.0 + TINY, math.cos(THETA)],
)
simulation.set_sources([source])

# =============================================================================
# Set tally
# =============================================================================
# Energy deposition tally along z-axis
z_bins = np.linspace(0.0, L, N_LAYERS + 1)
mesh = mcdc.MeshStructured(z=z_bins)

edep_tally = mcdc.Tally(name="edep", mesh=mesh, scores=["energy_deposition"])
flux_tally = mcdc.Tally(name="flux", scores=["flux"], mesh=mesh)
s1_current = mcdc.Tally(name="s1_current", surface=s1, scores=["current-net"])
s2_current = mcdc.Tally(name="s2_current", surface=s2, scores=["current-net"])
simulation.set_tallies([edep_tally, flux_tally, s1_current, s2_current])

# =============================================================================
# Settings and run
# =============================================================================

simulation.settings.set_transported_particles(["electron"])
simulation.settings.N_particle = N_PARTICLES
simulation.settings.active_bank_buffer = N_PARTICLES * 10

simulation.run()
