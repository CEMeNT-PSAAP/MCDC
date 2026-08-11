.. _example_iterative_source_reweighting:

============================
Iterative Source Reweighting
============================

This one-group fixed-source example demonstrates partial model updates across
several runs of one :class:`mcdc.Simulation`. A homogeneous slab contains
symmetric left and right sources. Each iteration changes only their relative
probabilities, compiles a complete new snapshot, and writes a separate output
file.

The three source mixtures are 20/80, 50/50, and 80/20. Because the geometry and
material are symmetric, the 20/80 and 80/20 flux profiles should approximately
mirror one another, while the 50/50 profile should be approximately symmetric
about the slab midpoint.

Full Input
----------

.. literalinclude:: ../../../examples/iterative_source_reweighting/input.py
   :language: python
   :linenos:

Post-processing
---------------

.. literalinclude:: ../../../examples/iterative_source_reweighting/process-output.py
   :language: python
   :linenos:

The post-processing script overlays the three spatial flux profiles, prints
the integrated flux in each half of the slab, and reports two symmetry
comparisons.

How to Run
----------

From inside ``examples/iterative_source_reweighting``:

.. code-block:: sh

   python input.py
   python process-output.py

The calculation writes ``source_mix_left_20.h5``,
``source_mix_left_50.h5``, and ``source_mix_left_80.h5``. Post-processing
writes ``iterative_source_comparison.png``.
