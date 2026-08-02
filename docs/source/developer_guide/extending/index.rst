.. _extending_mcdc:

===============
Extending MC/DC
===============

This section provides implementation-oriented guidance for adding new model concepts and transport capabilities to MC/DC.
It complements the :doc:`../architecture/index`, which explains the existing design, by describing the changes an extension must make across the Python model, compilation, generated runtime layers, transport implementation, public API, and tests.

Start with :doc:`extending_the_object_model` when adding a field to an existing model class, introducing simulation-owned configuration, creating a registered model-object category, or implementing a new polymorphic subtype.
Continue with :doc:`writing_numba_compatible_transport_code` when the extension adds or changes code executed during particle transport.

Use the :doc:`../../contributing/index` for repository setup, test commands, continuous-integration coverage, and pull-request requirements.

.. toctree::
   :maxdepth: 1

   extending_the_object_model
   writing_numba_compatible_transport_code
