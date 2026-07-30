.. _user_tallies:

=============================
Tallies and Post-processing
=============================

Tallies specify which transport quantities MC/DC records and how those
quantities are divided into bins. This page extends the model from
:doc:`getting_started/first_simulation`; add the examples below before
``simulation.run()``. The code
snippets assume NumPy has been imported as ``np``.

Each tally combines:

- one or more scores, such as flux, collision rate, or current;
- optional spatial, angular, energy, time, surface, or cell filters; and
- a name used to identify the tally in the output file.

Only tallies passed to ``simulation.set_tallies(...)`` are scored.

Mesh Tallies
------------

The introductory example scores flux over a structured z mesh:

.. code-block:: python3

   mesh = mcdc.MeshStructured(z=np.linspace(0.0, 6.0, 61))
   flux_tally = mcdc.Tally(
       name="slab_flux",
       mesh=mesh,
       scores=["flux"],
   )
   simulation.set_tallies([flux_tally])

The 61 grid points define 60 spatial bins. Track-length scores include
``"flux"``, ``"density"``, ``"collision"``, ``"capture"``, and
``"fission"``.

Angular Filters
---------------

Add polar-cosine boundaries to retain angular information:

.. code-block:: python3

   angular_flux_tally = mcdc.Tally(
       name="angular_flux",
       mesh=mesh,
       mu=np.linspace(-1.0, 1.0, 33),
       scores=["flux"],
   )
   simulation.set_tallies([angular_flux_tally])

The resulting tally contains 32 polar-angle bins in each spatial bin. A
reference direction can be supplied with ``polar_reference``; its default is
the positive z direction.

Surface-crossing Tallies
------------------------

A surface filter scores net current across a particular surface:

.. code-block:: python3

   interface_current = mcdc.Tally(
       name="interface_current",
       surface=interface,
       scores=["current-net"],
   )

The current sign follows the orientation of the surface normal. For the
``PlaneZ`` at the material interface, crossings toward increasing z contribute
positively and crossings toward decreasing z contribute negatively.

A cell filter can score current across every boundary of a cell:

.. code-block:: python3

   shield_cell_current = mcdc.Tally(
       name="shield_cell_current",
       cell=shield_cell,
       scores=["current-net", "current-in", "current-out"],
   )

``"current-in"`` and ``"current-out"`` are positive partial currents;
``"current-net"`` retains the crossing sign.

Combining cell and surface filters restricts the tally to one surface while
using the cell to classify incoming and outgoing crossings:

.. code-block:: python3

   shield_interface_current = mcdc.Tally(
       name="shield_interface_current",
       surface=interface,
       cell=shield_cell,
       scores=["current-net", "current-in", "current-out"],
   )

Attach every requested tally in one call:

.. code-block:: python3

   simulation.set_tallies(
       [
           flux_tally,
           interface_current,
           shield_cell_current,
           shield_interface_current,
       ]
   )

Statistical Uncertainty
-----------------------

MC/DC estimates tally uncertainty from statistically independent batches:

.. code-block:: python3

   simulation.settings.N_particle = 1_000
   simulation.settings.N_batch = 10

``N_particle`` is the number of histories per batch. Increasing
``N_particle`` reduces the noise within each batch, while ``N_batch`` controls
how many independent batch results contribute to the reported standard
deviation. At least two batches are required for a nonzero estimate.

Reading Tally Output
--------------------

Named tallies are stored under ``tallies/<name>`` in the output HDF5 file.
The filter grids and score results are stored below that group:

.. code-block:: text

   tallies/
     slab_flux/
       grid/
         z
       flux/
         mean
         sdev

Load and normalize the mesh flux with h5py:

.. code-block:: python3

   import h5py

   with h5py.File("slab_shielding.h5", "r") as output:
       tally = output["tallies/slab_flux"]
       z = tally["grid/z"][:]
       flux = tally["flux/mean"][:]
       flux_sdev = tally["flux/sdev"][:]

   dz = z[1:] - z[:-1]
   flux /= dz
   flux_sdev /= dz

The score arrays contain values integrated over their bins. Divide by the
applicable spatial, angular, energy, or time widths when a differential result
is required.

Reducing an Angular Tally
-------------------------

For the angular tally above, sum the angle-bin contributions to recover scalar
flux. Weighting by the polar-cosine midpoint gives a midpoint approximation of
the z-directed current:

.. code-block:: python3

   with h5py.File("slab_shielding.h5", "r") as output:
       tally = output["tallies/angular_flux"]
       z = tally["grid/z"][:]
       mu = tally["grid/mu"][:]
       angular_flux = tally["flux/mean"][:]
       angular_flux_sdev = tally["flux/sdev"][:]

   dz = z[1:] - z[:-1]
   dmu = mu[1:] - mu[:-1]
   mu_mid = 0.5 * (mu[:-1] + mu[1:])

   scalar_flux = np.sum(angular_flux, axis=0) / dz
   scalar_flux_sdev = np.linalg.norm(angular_flux_sdev, axis=0) / dz
   current = np.sum(
       angular_flux * mu_mid[:, np.newaxis],
       axis=0,
   ) / dz

The exact array-axis order follows the active filters and is recorded by the
corresponding grids in the tally group. Inspect the output shapes before
performing reductions.

Verification
------------

Before drawing conclusions from a tally:

- Confirm that its filters cover the intended phase-space region.
- Check that the result changes consistently when the mesh is refined.
- Increase the particle population and verify that uncertainty decreases.
- Use current tallies to check particle balance where appropriate.
- Compare against an analytic or benchmark solution when one is available.

The :class:`mcdc.Tally` API reference documents all supported scores and filter
combinations.
