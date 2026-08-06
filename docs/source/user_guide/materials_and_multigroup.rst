.. _user_materials_and_multigroup:

========================================
Materials and Neutron Multigroup Data
========================================

A :class:`mcdc.Material` may contain a native nuclide or element composition,
neutron multigroup data, or both.

Native Materials
----------------

Define a native material with either nuclide or element atomic densities:

.. code-block:: python

   fuel = mcdc.Material(
       name="UO2",
       nuclide_composition={
           "U235": 5.0e-4,
           "U238": 2.2e-2,
           "O16": 4.5e-2,
       },
       temperature=293.6,
   )

Multigroup Materials
--------------------

:class:`mcdc.NeutronMultigroupData` stores macroscopic neutron cross sections,
group speeds, fission spectra, and delayed-precursor data. For a material that
uses only this transport model, :meth:`mcdc.Material.multigroup` is the concise
entry point:

.. code-block:: python

   moderator = mcdc.Material.multigroup(
       name="Moderator",
       capture=np.array([0.1, 0.2]),
       scatter=np.array([
           [0.7, 0.1],
           [0.2, 0.5],
       ]),
       energy_grid=np.array([1.0e-5, 1.0, 20.0e6]),
   )

Energy Grids and Representation
-------------------------------

An explicit ``energy_grid`` contains ``G + 1`` boundaries. Group ``g`` covers
``energy_grid[g] <= E < energy_grid[g + 1]``. The grid both maps continuous
energy to a group and bounds continuous energy reconstructed from a group.

When no grid is supplied, MC/DC creates the group-coordinate grid
``[1.0e-6 - 0.5, 0.5, 1.5, 2.5, ...]`` with
``energy_representation="midpoint"``. An explicit grid also supports
``"log_midpoint"``, ``"uniform"``, and ``"log_uniform"`` reconstruction.

Hybrid Materials
----------------

A material can combine native and neutron multigroup data:

.. code-block:: python

   hybrid_fuel = mcdc.Material(
       name="Hybrid fuel",
       nuclide_composition={"U235": 5.0e-4, "U238": 2.2e-2},
       neutron_multigroup=mcdc.NeutronMultigroupData(
           capture=np.array([0.10]),
           fission=np.array([0.20]),
           nu_p=np.array([2.50]),
           energy_grid=np.array([1.0e-5, 20.0e6]),
       ),
   )

The explicit energy grid is required whenever a native composition and neutron
multigroup data are combined. Its bounds identify the energy interval where
the multigroup transport model is available.

Shared and Material-local Grids
-------------------------------

Neutron multigroup datasets use one shared energy grid by default. Enable the
multigrid option for material-local group structures:

.. code-block:: python

   simulation.technique.neutron_multigroup(multigrid=True)

Source Energy and Group
-----------------------

A particle carries continuous ``energy`` and an integer ``group`` as separate
state. :class:`mcdc.Source` accepts an independent continuous energy
distribution and discrete group distribution:

.. code-block:: python

   source = mcdc.Source(
       energy=1.0e6,
       group=([0, 1], [0.25, 0.75]),
   )

``group`` is a general transport-mode state; neutron multigroup transport uses
it as the neutron energy-group index. If both variables are supplied, the
group takes precedence over energy only for shared-grid neutron multigroup
transport. The tally equivalents are the separate ``group`` and ``energy``
filters described in :doc:`tallies`.
