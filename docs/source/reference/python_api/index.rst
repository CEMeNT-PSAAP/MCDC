.. _python_api:

==========
Python API
==========

The MC/DC public API is centered on :class:`mcdc.Simulation`. A simulation owns
the model geometry and material, sources, tallies, settings, transport
techniques, and runtime state needed for one calculation.

Build the model with the public objects listed below, attach its root cells,
sources, and tallies to a simulation, and then visualize or run that simulation:

.. code-block:: python

   simulation = mcdc.Simulation(name="Example")
   simulation.set_model([cell])
   simulation.set_sources([source])
   simulation.set_tallies([tally])
   simulation.settings.N_particle = 10_000
   simulation.run()

The complete public interfaces and additional examples are documented on each
linked API page.


Simulation
----------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: simulationclass.rst

   mcdc.Simulation


Model building blocks
---------------------

Materials
^^^^^^^^^

Materials define the interaction data used by cells. Use
:class:`mcdc.Material` for continuous-energy transport and
:class:`mcdc.MaterialMG` for multigroup transport.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: omcclass.rst

   mcdc.Material
   mcdc.MaterialMG


Geometry
^^^^^^^^

Surfaces bound spatial regions, cells pair those regions with materials or
universes, and universes and lattices organize repeated geometry.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: omcclass.rst

   mcdc.Surface
   mcdc.Cell
   mcdc.Universe
   mcdc.Lattice


Sources
^^^^^^^

Sources describe the initial particle population.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: omcclass.rst

   mcdc.Source


Tallies
^^^^^^^

Tallies define the quantities to score and the filters over which those scores
are accumulated.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: omcclass.rst

   mcdc.Tally


Meshes
^^^^^^

Meshes provide spatial bins for mesh-filtered tallies and transport
techniques.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: omcclass.rst

   mcdc.MeshUniform
   mcdc.MeshStructured


Configuration and execution
---------------------------

Simulation settings
^^^^^^^^^^^^^^^^^^^

Each :class:`mcdc.Simulation` owns its settings at ``simulation.settings``.
Settings control particle histories, batches, random-number generation,
transport modes, census times, particle banks, output, and GPU execution.
Specialized modes are configured through methods such as
``simulation.settings.set_eigenmode(...)`` and
``simulation.settings.set_time_census(...)``. See :class:`mcdc.Simulation` for
the complete settings interface.

Transport techniques
^^^^^^^^^^^^^^^^^^^^

Transport techniques are configured directly on the simulation instance. See
:class:`mcdc.Simulation` for their signatures and examples.

Compiling and running
^^^^^^^^^^^^^^^^^^^^^

Calling ``simulation.run()`` compiles the current Python object graph when
needed, executes particle transport, and writes the configured output.
``simulation.visualize_model(...)`` similarly compiles when needed before
rendering the model. Use ``simulation.compile()`` when an explicit compiled
snapshot is required before either operation.
