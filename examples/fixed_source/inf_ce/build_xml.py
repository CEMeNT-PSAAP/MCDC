from math import log10

import numpy as np
import openmc

###############################################################################
# Create materials for the problem

uo2 = openmc.Material(name='UO2 fuel at 2.4% wt enrichment')
uo2.set_density('g/cm3', 10.29769)
uo2.add_element('U', 1., enrichment=2.4)
uo2.add_element('O', 2.)

borated_water = openmc.Material(name='Borated water')
borated_water.set_density('g/cm3', 1.0)
borated_water.add_element('B', 5e-4)
borated_water.add_element('H', 5.0e-2)
borated_water.add_element('O', 2.4e-2)
borated_water.add_s_alpha_beta('c_H_in_H2O')

water = openmc.Material(name='Water')
water.set_density('g/cm3', 1.0)
water.add_element('H', 2.0)
water.add_element('O', 1.0)
water.add_s_alpha_beta('c_H_in_H2O')

#print(uo2.get_nuclide_atom_densities())
#print(borated_water.get_nuclide_atom_densities())
#print(water.get_nuclide_atom_densities())

# Collect the materials together and export to XML
materials = openmc.Materials([uo2, borated_water, water])
materials.export_to_xml()

###############################################################################
# Define problem geometry

# Create cylindrical surfaces
clad_or = openmc.ZCylinder(r=0.45720, name='Clad OR')

# Create a region represented as the inside of a rectangular prism
pitch = 1.25984
box = openmc.rectangular_prism(pitch, pitch, boundary_type='reflective')

# Create cells, mapping materials to regions
fuel = openmc.Cell(fill=water, region=-clad_or)
water = openmc.Cell(fill=water, region=+clad_or & box)

# Create a geometry and export to XML
geometry = openmc.Geometry([fuel, water])
geometry.export_to_xml()

###############################################################################
# Define problem settings

# Indicate how many particles to run
settings = openmc.Settings()
settings.run_mode = 'fixed source'
settings.particles = 100000
settings.batches = 100

# Create an initial uniform spatial source distribution over fissionable zones
lower_left = (-pitch/2, -pitch/2, -1)
upper_right = (pitch/2, pitch/2, 1)
uniform_dist = openmc.stats.Box(lower_left, upper_right)
energy = openmc.stats.Uniform(1e6-1, 1e6+1)
settings.source = openmc.IndependentSource(space=uniform_dist,energy=energy)
settings.export_to_xml()

###############################################################################
# Define tallies

# Create a mesh filter that can be used in a tally
time_filter = openmc.TimeFilter(np.insert(np.logspace(-8, 2, 50), 0, 0.0))
with np.load("SHEM-361.npz") as data:
    E = data["E"]
energy_filter = openmc.EnergyFilter(E)

# Now use the mesh filter in a tally and indicate what scores are desired
tally = openmc.Tally(name="TD spectrum")
tally.filters = [time_filter, energy_filter]
tally.scores = ['flux']

# Instantiate a Tallies collection and export to XML
tallies = openmc.Tallies([tally])
tallies.export_to_xml()
