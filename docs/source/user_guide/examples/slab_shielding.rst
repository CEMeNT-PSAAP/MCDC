.. _example_slab_shielding:

==============
Slab Shielding
==============

This one-group fixed-source problem introduces the complete MC/DC workflow
with two materials, two slab cells, an isotropic source, and a mesh flux
tally. The :doc:`../getting_started/first_simulation` guide explains the input
step by step.

Full Input
----------

.. literalinclude:: ../../../../examples/slab_shielding/input.py
   :language: python
   :linenos:

Post-processing
---------------

.. literalinclude:: ../../../../examples/slab_shielding/process-output.py
   :language: python
   :linenos:

How to Run
----------

From inside ``examples/slab_shielding``:

.. code-block:: sh

   python input.py
   python process-output.py

The transport calculation writes ``slab_shielding.h5``. The
post-processing script reads its flux tally and writes
``slab_shielding_flux.png``.
