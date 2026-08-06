.. _theory:

==================
Theory and Methods
==================

Theory and Methods explains the physical models, mathematical formulations, and
numerical algorithms implemented in MC/DC. Use these pages to understand why a
method works; use the :doc:`../user_guide/index` to learn how to configure it
and the :doc:`../developer_guide/index` for software implementation details.

New to Monte Carlo transport? Start with :ref:`monte_carlo` for the
fundamentals and :ref:`geometry` for how MC/DC represents problem domains.
Then explore the advanced topics below.

For an additional external resource, see the
`OpenMC theory guide <https://docs.openmc.org/en/latest/methods/index.html>`_.

Fundamentals
------------

.. toctree::
   :maxdepth: 1

   monte_carlo
   geometry
   k_eigenvalue

Advanced Methods
----------------

.. toctree::
   :maxdepth: 1

   variance_reduction
   iqmc
   weight_windows
   uncertainty_quantification
   compressed_sensing

Transport Models
----------------

.. toctree::
   :maxdepth: 1

   continuous_energy
   domain_decomposition
   continuous_movement
