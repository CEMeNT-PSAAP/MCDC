.. _extending_mcdc:

===============
Extending MC/DC
===============

Use this section to extend the Python model, compilation, generated runtime layers, transport implementation, public API, and tests.
Read :doc:`../architecture/index` for the design that these recipes extend.

Start with :doc:`extending_the_object_model` when adding a field to an existing model class, introducing simulation-owned configuration, creating a registered model-object category, or implementing a new polymorphic subtype.
Continue with :doc:`writing_numba_compatible_transport_code` when the extension adds or changes code executed during particle transport.

For example, a new runtime field with no transport behavior uses the object-model guide, a numerical change using existing fields starts with the transport-code guide, and a new tally subtype follows both in that order.

Use the :doc:`../../contributing/index` for repository setup, test commands, continuous-integration coverage, and pull-request requirements.

.. toctree::
   :maxdepth: 1

   extending_the_object_model
   writing_numba_compatible_transport_code
