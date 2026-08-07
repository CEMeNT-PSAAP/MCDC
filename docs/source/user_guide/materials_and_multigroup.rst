.. _user_materials_and_multigroup:

============================
Materials and Transport Data
============================

A :class:`mcdc.Material` describes a physical medium. A nuclide or element
composition establishes its :ref:`native transport data <user_native_transport>`.
Materials can also carry particle-specific data that augments native
interaction data, for example with semi-empirical information, or supports
specialized and reduced transport treatments.

.. _user_native_transport:

Native Composition and Data
---------------------------

In MC/DC, *native* refers to data-library-backed transport physics derived from
a material's nuclide or element composition. The composition identifies the
physical constituents whose library records provide the interaction data.

Define a native material with either nuclide or element atomic densities in
atoms/(barn cm). Material temperature is specified in K:

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

Particle-specific Transport Data
--------------------------------

Particle-specific transport data is attached to the material that uses it. The
transport physics determines when and how each dataset contributes to a
particle interaction.

Neutron Multigroup Transport
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neutron multigroup transport represents neutron energy with discrete groups
and describes interactions using groupwise macroscopic data. It is widely used
in general transport pedagogy and in nuclear engineering applications.
:meth:`mcdc.Material.multigroup` creates a material with
:class:`mcdc.NeutronMultigroupData`, which holds the cross sections, group
speeds, production spectra, and delayed-precursor data. Macroscopic cross
sections use cm\ :sup:`-1`, group speeds use cm/s, and precursor decay rates use
s\ :sup:`-1`:

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

Multigroup Energy Grids and Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An explicit ``energy_grid`` contains ``G + 1`` physical energy boundaries in
eV. Group ``g`` covers ``energy_grid[g] <= E < energy_grid[g + 1]``. The grid
both maps continuous energy to a group and bounds continuous energy
reconstructed from a group.

When no grid is supplied, MC/DC creates the group-coordinate grid
``[-0.5, 0.5, 1.5, 2.5, ...]`` with
``energy_representation="midpoint"``. An explicit grid also supports
``"log_midpoint"``, ``"uniform"``, and ``"log_uniform"`` reconstruction.

Neutron multigroup datasets use one shared energy grid by default. Enable the
multigrid option when materials use different group structures:

.. code-block:: python

   simulation.technique.neutron_multigroup(multigrid=True)

Combining Native and Multigroup Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct :class:`mcdc.NeutronMultigroupData` directly when attaching it to a
material with a :ref:`native composition <user_native_transport>`:

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

Particle Energy and Group State
-------------------------------

A particle carries continuous ``energy`` and an auxiliary integer ``group`` as
separate state. :class:`mcdc.Source` accepts an independent continuous energy
distribution in eV and discrete group distribution:

.. code-block:: python

   source = mcdc.Source(
       energy=1.0e6,
       group=([0, 1], [0.25, 0.75]),
   )

The physics using ``group`` determines its meaning. Neutron multigroup
transport uses it as the neutron energy-group index. If both variables are
supplied, the group takes precedence over energy only for shared-grid neutron
multigroup transport. The tally equivalents are the separate ``group`` and
``energy`` filters described in :doc:`tallies`.
