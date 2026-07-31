.. _example_validation:

==================
Example Validation
==================

The inputs under ``examples/`` are executable documentation. Changes to the
public API, object-compilation behavior, or example models should validate them
at three levels.

Compile Every Example
---------------------

The automated example test executes every ``examples/**/input.py`` while
replacing transport and visualization with model compilation:

.. code-block:: sh

   python -m pytest test/unit/test_example_inputs.py --mode=python

This check:

- Discovers new example inputs automatically.
- Executes their model-construction code from the correct working directory.
- Requires an explicit :class:`mcdc.Simulation`.
- Allows iterative examples to compile the same simulation more than once.
- Compiles the complete reachable object graph.
- Requires at least one cell, source, and tally.
- Avoids long particle-transport runs and output files.

The test catches removed public interfaces, missing object attachments,
invalid model references, compilation failures, and example files that no
longer reach ``simulation.run()`` or ``simulation.visualize_model()``.

Run Representative Transport
----------------------------

The compile-only test does not validate transport results. Before merging a
change that affects execution, run the small slab-shielding problem in the
affected CPU modes:

.. code-block:: sh

   cd examples/slab_shielding
   python input.py --mode=python --N_particle=100 --N_batch=2
   python input.py --mode=numba --N_particle=100 --N_batch=2

Check that both commands finish, write ``slab_shielding.h5``, and produce the
expected ``slab_flux`` tally. Use the normal example settings when evaluating
statistical agreement rather than only execution.

Validate Specialized Paths
--------------------------

Run examples that represent the changed capability:

- ``examples/c5g7/k-eigenvalue`` for k-eigenvalue behavior.
- ``examples/moving_source`` or ``examples/c5g7/transient`` for transient
  behavior.
- ``examples/slab_shielding`` or ``examples/fuel_array_packaged`` for
  visualization.
- A supported accelerator environment for Numba-GPU changes.

Regression tests remain the authoritative check for numerical results.
Example validation complements them by ensuring the documented, user-facing
inputs continue to construct models through the current public API.
