.. _simulation_lifecycle:

====================
Simulation Lifecycle
====================

An MC/DC calculation is owned by a :class:`mcdc.Simulation`. The simulation
collects one model, its sources and tallies, its settings and techniques, and
the runtime state needed to visualize or execute it.

The normal workflow has five stages:

#. Construct model objects.
#. Attach the model roots to a simulation.
#. Compile the connected object graph.
#. Visualize or run the prepared model.
#. Generate and post-process output.

Most inputs call only ``visualize_model`` or ``run`` explicitly. MC/DC performs
the required compilation and runtime preparation automatically.

1. Construct Model Objects
--------------------------

Materials, surfaces, cells, sources, meshes, and tallies are ordinary Python
objects:

.. code-block:: python

   material = mcdc.Material.multigroup(
       capture=np.array([0.1]),
       scatter=np.array([[0.9]]),
   )
   left = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
   right = mcdc.Surface.PlaneZ(z=2.0, boundary_condition="vacuum")
   cell = mcdc.Cell(region=+left & -right, fill=material)

Objects can refer to other objects. In this example the cell retains its
region, the region retains its surfaces, and the cell retains its material.

2. Attach Roots to a Simulation
-------------------------------

Create a simulation and attach the roots from which MC/DC can discover the
complete model:

.. code-block:: python

   simulation = mcdc.Simulation("Slab")
   simulation.set_model([cell])
   simulation.set_sources([source])
   simulation.set_tallies([tally])

``set_model`` accepts the cells in the root universe. Materials, surfaces,
child universes, lattices, and meshes that are reachable from those cells do
not need separate registration calls.

Sources and tallies are explicit roots because they are not necessarily
reachable from the geometry. Settings and techniques belong directly to the
simulation:

.. code-block:: python

   simulation.settings.N_particle = 10_000
   simulation.settings.N_batch = 20
   simulation.technique.implicit_capture()

This explicit ownership replaces the former process-wide singleton workflow.

Process Ownership
^^^^^^^^^^^^^^^^^

MC/DC runs one active simulation context at a time within a Python process.
Objects may be shared within one model—for example, several cells may use the
same material—but the same object instance should not be attached to different
``Simulation`` instances.

Multiple simulations can be constructed in one process when they have
independent object graphs and are run serially. For concurrent calculations,
construct each model and run its simulation in a separate process. A Python
driver, workflow system, or batch scheduler can manage those processes.

3. Compile the Object Graph
---------------------------

Compilation discovers every object reachable from the simulation, registers
shared objects once, and assigns simulation-local IDs. Normally it is
automatic:

- ``simulation.visualize_model(...)`` compiles when needed.
- ``simulation.run()`` compiles when needed.

Call ``simulation.compile()`` directly only when you need to inspect the
compiled object lists or IDs before visualization or execution:

.. code-block:: python

   simulation.compile()
   print(simulation.cells)
   print(material.ID)

Compiled IDs describe one snapshot and may change after recompilation. Do not
use them as persistent identifiers in an input or post-processing workflow.

The three setter methods invalidate the current snapshot. If you directly
modify an attached object after explicitly compiling or visualizing the model,
call ``simulation.compile()`` again before inspecting or visualizing the
change.

Compilation finalizes some user-facing values in place.
In particular, source probabilities are normalized, tally limits may reduce ``settings.time_boundary``, and particle-bank buffer ratios may be adjusted for the selected run mode.
Set new raw values explicitly before recompiling when an iterative workflow changes one of these inputs.

4. Visualize or Run
-------------------

Visualization is a useful geometry check before transport:

.. code-block:: python

   simulation.visualize_model(
       vis_plane="xz",
       x=[-1.0, 1.0],
       y=0.0,
       z=[0.0, 2.0],
       pixels=(100, 200),
       colors=None,
       time=[0.0],
       save_as="slab_geometry",
   )

Run transport after validating the model:

.. code-block:: python

   simulation.run()

Begin in Python mode:

.. code-block:: sh

   python input.py --mode=python

After the model and workflow are correct, enable Numba-CPU:

.. code-block:: sh

   python input.py --mode=numba

GPU execution adds another acceleration and runtime layer:

.. code-block:: sh

   python input.py --mode=numba --target=gpu

See :doc:`execution/cpu` and :doc:`execution/gpu` for operational guidance.

5. Process Output
-----------------

``simulation.run()`` writes the configured HDF5 output. Use stable tally names
and a companion post-processing script so the relationship between an input
and its analysis remains clear:

.. code-block:: python

   tally = mcdc.Tally(
       name="slab_flux",
       mesh=mesh,
       scores=["flux"],
   )
   simulation.set_tallies([tally])
   simulation.settings.output_name = "slab"

The tally is then available under ``tallies/slab_flux`` in ``slab.h5``.

For workflows that reuse most of a model while changing selected inputs between
runs, continue with :doc:`iterative_simulations`.

Complete Workflows
------------------

The example suite demonstrates the lifecycle in complete inputs:

- :ref:`example_slab_shielding` — fixed-source execution, model
  visualization, and output processing.
- :ref:`example_iterative_source_reweighting` — partial model updates and
  repeated compilation within one simulation.
- :ref:`example_moving_source` — transient fixed-source transport.
- :ref:`example_c5g7_k_eigenvalue` — k-eigenvalue execution.
- :ref:`example_c5g7_transient` — a larger reactor transient.

For exact public signatures, use the :doc:`../reference/python_api/index`.
For framework implementation details, continue with
:doc:`../developer_guide/architecture/simulation_compilation` and
:doc:`../developer_guide/architecture/runtime_data_layout`.
