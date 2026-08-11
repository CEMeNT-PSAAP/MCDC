import numpy as np
import mcdc

simulation = mcdc.Simulation("Sphere in cube")

# ======================================================================================
# Set model
# ======================================================================================
# Homogeneous pure-fission sphere inside a pure-scattering cube

# Set materials
pure_f = mcdc.Material.multigroup(fission=np.array([1.0]), nu_p=np.array([1.2]))
pure_s = mcdc.Material.multigroup(scatter=np.array([[1.0]]))

# Set surfaces
sx1 = mcdc.Surface.PlaneX(x=0.0, boundary_condition="vacuum")
sx2 = mcdc.Surface.PlaneX(x=4.0, boundary_condition="vacuum")
sy1 = mcdc.Surface.PlaneY(y=0.0, boundary_condition="vacuum")
sy2 = mcdc.Surface.PlaneY(y=4.0, boundary_condition="vacuum")
sz1 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
sz2 = mcdc.Surface.PlaneZ(z=4.0, boundary_condition="vacuum")
sphere = mcdc.Surface.Sphere(center=[2.0, 2.0, 2.0], radius=1.5)
inside_sphere = -sphere
inside_box = +sx1 & -sx2 & +sy1 & -sy2 & +sz1 & -sz2

# Set cells
box_cell = mcdc.Cell(name="Box cover", region=inside_box & ~inside_sphere, fill=pure_s)
sphere_cell = mcdc.Cell(name="The sphere", region=inside_sphere, fill=pure_f)
simulation.set_model([box_cell, sphere_cell])

# ======================================================================================
# Set source
# ======================================================================================

source = mcdc.Source(
    x=[0.0, 4.0],
    y=[0.0, 4.0],
    z=[0.0, 4.0],
    isotropic=True,
    energy=0,
    time=[0.0, 50.0],
)
simulation.set_sources([source])

# ======================================================================================
# Set tallies, settings, techniques, and run MC/DC
# ======================================================================================

# Tallies
tally = mcdc.Tally(
    name="Spherical fission detector", cell=sphere_cell, scores=["fission"]
)
simulation.set_tallies([tally])

# Settings
simulation.settings.N_particle = 1000
simulation.settings.N_batch = 2

# Techniques
simulation.technique.implicit_capture()

# Run
simulation.run()
