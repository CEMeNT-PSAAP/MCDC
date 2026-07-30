.. _architecture:

============
Architecture
============

Architecture documentation explains how MC/DC translates flexible Python model
definitions into high-performance particle-transport execution.

Start with :doc:`acceleration` for an overview of the acceleration and code
generation layers. Read :doc:`gpu` when working on accelerator execution or
the Harmonize integration.

Documentation of the explicit ``Simulation`` compilation pipeline, runtime data
layout, and Numba design belongs in this section as those pages are developed.

.. toctree::
   :maxdepth: 1

   acceleration
   gpu
