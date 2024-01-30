import openmc

rho_uo2 = 10.97
rho_h2o = 0.997
rho_b4c = 2.52

a_u235 = 0.05
a_u238 = 1.0 - a_u235

A_uo2 = 270.03
A_h2o = 18.01528
A_b4c = 55.255

N_avo = 6.023e23

N_uo2 = rho_uo2 * N_avo / A_uo2 * 1e-24
N_h2o = rho_h2o * N_avo / A_h2o * 1e-24
N_b4c = rho_b4c * N_avo / A_b4c * 1e-24

N_u235 = a_u235 * N_uo2
N_u238 = a_u238 * N_uo2
N_o16_uo2 = 2.0 * N_uo2

N_mix = N_u235 + N_u238 + N_o16_uo2
N_u235 = N_u235 / N_mix
N_u238 = N_u238 / N_mix
N_o16_uo2 = N_o16_uo2 / N_mix

uo2 = openmc.Material()
uo2.set_density("atom/b-cm", N_mix)
uo2.add_nuclide("U235", N_u235)
uo2.add_nuclide("U238", N_u238)
uo2.add_nuclide("O16", N_o16_uo2)

N_h1 = 2.0 * N_h2o
N_o16_h2o = N_h2o

N_mix = N_h1 + N_o16_h2o
N_h1 = N_h1 / N_mix
N_o16_h2o = N_o16_h2o / N_mix

h2o = openmc.Material()
h2o.set_density("atom/b-cm", N_mix)
h20.add_nuclide("H1", N_h1)
h20.add_nuclide("O16", N_o16_h2o)

N_b10 = 4.0 * N_b4c
N_c12 = N_b4c

N_mix = N_b10 + N_c12
N_b10 = N_b10 / N_mix
N_c12 = N_c12 / N_mix

b4c = openmc.Material()
b4c.set_density("atom/b-cm", N_mix)
b4c.add_nuclide("B10", N_b10)
b4c.add_nuclide("C12", N_c12)

mats = openmc.Materials([h20, b4c, uo2])
mats.export_to_xml()

# Create a 5 cm x 5 cm box filled with iron
box = openmc.model.rectangular_prism(10.0, 10.0, boundary_type="vacuum")
cell = openmc.Cell(fill=iron, region=box)
geometry = openmc.Geometry([cell])
geometry.export_to_xml()

# Tell OpenMC we're going to use our custom source
settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.batches = 10
settings.particles = 1000
source = openmc.CompiledSource()
source.library = "build/libsource.so"
settings.source = source
settings.export_to_xml()

# Finally, define a mesh tally so that we can see the resulting flux
mesh = openmc.RegularMesh()
mesh.lower_left = (-5.0, -5.0)
mesh.upper_right = (5.0, 5.0)
mesh.dimension = (50, 50)

tally = openmc.Tally()
tally.filters = [openmc.MeshFilter(mesh)]
tally.scores = ["flux"]
tallies = openmc.Tallies([tally])
tallies.export_to_xml()
