import numpy as np
import mcdc

# Create MC/DC simulation
simulation = mcdc.Simulation("Cooper problem 2")

# =============================================================================
# Set model
# =============================================================================
# A shielding problem based on Problem 2 of [Coper NSE 2001]
# https://ans.tandfonline.com/action/showCitFormats?doi=10.13182/NSE00-34

# Set materials
SigmaT = 5.0
c = 0.8
m_barrier = mcdc.Material.multigroup(
    capture=np.array([SigmaT]), scatter=np.array([[SigmaT * c]])
)
SigmaT = 1.0
m_room = mcdc.Material.multigroup(
    capture=np.array([SigmaT]), scatter=np.array([[SigmaT * c]])
)

# Set surfaces
sx1 = mcdc.Surface.PlaneX(x=0.0, boundary_condition="reflective")
sx2 = mcdc.Surface.PlaneX(x=2.0)
sx3 = mcdc.Surface.PlaneX(x=2.4)
sx4 = mcdc.Surface.PlaneX(x=4.0, boundary_condition="vacuum")
sy1 = mcdc.Surface.PlaneY(y=0.0, boundary_condition="reflective")
sy2 = mcdc.Surface.PlaneY(y=2.0)
sy3 = mcdc.Surface.PlaneY(y=4.0, boundary_condition="vacuum")

# Set cells
room_lower_left = mcdc.Cell(region=+sx1 & -sx2 & +sy1 & -sy2, fill=m_room)
room_upper = mcdc.Cell(region=+sx1 & -sx4 & +sy2 & -sy3, fill=m_room)
room_lower_right = mcdc.Cell(region=+sx3 & -sx4 & +sy1 & -sy2, fill=m_room)
barrier = mcdc.Cell(region=+sx2 & -sx3 & +sy1 & -sy2, fill=m_barrier)
simulation.set_model([room_lower_left, room_upper, room_lower_right, barrier])

# =============================================================================
# Set source
# =============================================================================

source = mcdc.Source(
    x=[0.0, 1.0],
    y=[0.0, 1.0],
    isotropic=True,
    energy_group=0,
    time=0.0,
)
simulation.set_sources([source])

# =============================================================================
# Set tallies, settings, techniques, and run MC/DC
# =============================================================================

# Tallies
mesh = mcdc.MeshUniform(x=(0.0, 0.1, 40), y=(0.0, 0.1, 40))
tally = mcdc.Tally(mesh=mesh, scores=["flux"])
simulation.set_tallies([tally])

# Settings
simulation.settings.N_particle = 1000
simulation.settings.N_batch = 2

# Techniques
simulation.implicit_capture()
simulation.global_weight_roulette(0.1, 1.0)

# Run
simulation.run()
