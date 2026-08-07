.. _contributing:

============
Contributing
============

This guide describes the repository workflow and quality checks for contributing to MC/DC.
It is intended for both occasional contributors and project maintainers.

Start with the setup steps below.
Use :doc:`continuous_integration` to understand automated checks and :doc:`container_development` when developing in the project container.
Use :doc:`example_validation` when changing the public API or example problems.
Read :doc:`pull_requests` before preparing a contribution.
For software architecture and documentation practices, see the :doc:`../developer_guide/index`.

For implementation guidance specific to compiled transport functions, see :doc:`../developer_guide/extending/writing_numba_compatible_transport_code`.

Contributions target the ``dev`` branch. To prepare a development checkout:

#. Fork ``mcdc-project/mcdc`` to your GitHub account.
#. ``git clone git@github.com:<YOUR_GITHUB>/mcdc.git``
#. ``git switch dev``
#. Run the installation script to install MC/DC as an editable package.

Development Workflow
--------------------

.. toctree::
   :maxdepth: 1

   continuous_integration
   container_development
   example_validation
   pull_requests

MC/DC documentation is an important part of the project and evolves alongside the codebase.
The :doc:`../developer_guide/documentation/index` guide describes the documentation philosophy, writing guidelines, and the tools used to build and maintain the documentation.

Please note our `code of conduct <https://github.com/mcdc-project/mcdc/blob/dev/CODE_OF_CONDUCT.md>`_, which we take seriously.

------------
Code Styling
------------

Our code is auto-linted for the `Black code style <https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html>`_.
Your contributions will not be merged unless you follow this code style.
It's pretty easy to do this locally, just run,

.. code-block:: sh


    pip install black
    black .


in the top level MC/DC directory and all necessary changes will be automatically made for you.

---------
Debugging
---------

MCDC includes options to debug the Numba JIT code.
It does this by toggling Numba options using the numba.config submodule.
This will result in less performant code and longer compile times but will allow for better error messages from Numba and other packages.
`See Numba documentation of a list of all possible debug and compiler options. <https://numba.readthedocs.io/en/stable/reference/envvars.html#debugging>`_
The most useful set of debug options for MC/DC can be enabled with

.. code-block:: python3

    python input.py --mode=numba_debug

Which will toggle the following debug and compiler options in Numba:

* ``DISABLE_JIT=False`` turns on the jitter
* ``NUMBA_OPT=0`` Forces the compilers to form un-optimized code (other options for this are ``1``, ``2``, and ``3`` with ``3`` being the most optimized). This option might need to be changed if errors only result from more optimization.
* ``DEBUG=False`` turns on all debugging options. This is still disabled in ``mcdc numba_debug`` as it will print ALOT of info on your terminal screen
* ``NUMBA_FULL_TRACEBACKS=1`` allows errors from sub-packages to be printed (i.e. Numpy)
* ``NUMBA_BOUNDSCHECK=1`` numba will check vectors for bounds errors. If this is disabled it bound errors will result in a ``seg_fault``. This in consort with the previous option allows for the exact location of a bound error to be printed from Numpy subroutines
* ``NUMBA_DEBUG_NRT=1`` enables the `Numba run time (NRT) statistics counter <https://numba.readthedocs.io/en/stable/developer/numba-runtime.html>`_ This helps with debugging memory leaks.
* ``NUMBA_DEBUG_TYPEINFER= 1`` print out debugging information about type inferences that numba might need to make if a function is ill-defined
* ``NUMBA_ENABLE_PROFILING=1`` enables profiler use
* ``NUMBA_DUMP_CFG=1`` prints out a control flow diagram

If extra debug options or alteration to these options are required they can be toggled and passed under the ``mode==numba_debug`` option tree in ``mcdc/config.py``.

-------
Caching
-------

MC/DC is a just-in-time (JIT) compiled code.
This is sometimes disadvantageous, especially for users who might run many versions of the same simulation with slightly different parameters.
As the JIT compilation scheme will only compile functions that are actually used in a given simulation, it is not a grantee that any one function will be compiled.

Developers should be very cautious about using caching features.
Numba has a few documented errors around caching.
The most critical of which is that functions in other files that are called by cached functions will not force a recompile, even if there are changes in those sub-functions.
In this case caching should be disabled.

In MC/DC the simulation functions (in ``mcdc/transport/simulation.py``) can be configured to use caching.
Caching behavior is controlled via the ``--caching`` and ``--clear_cache`` command-line flags.

To disable caching, omit the ``--caching`` flag (the default).
Alternatively a developer could delete the ``__pycache__`` directory or other cache directory which is system dependent (`see more about clearing the numba cache <https://numba.readthedocs.io/en/stable/developer/caching.html>`_)


At some point MC/DC will enable `Numba's Ahead of Time compilation abilities <https://numba.readthedocs.io/en/stable/user/pycc.html>`_. But the core development team is holding off until scheduled `upgrades to AOT functionality in Numba are implemented <https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-numba-pycc>`_.
However if absolutely required by users numba does allow for some `cache sharing <https://numba.readthedocs.io/en/stable/developer/caching.html>`_.

------------------
Adding a New Input
------------------

For architectural guidance on adding a model field, embedded configuration, registered object category, or polymorphic subtype, see :doc:`../developer_guide/extending/extending_the_object_model`.
Public model classes and configuration are primarily defined in ``mcdc/object_/``.
Common input-related locations include:

#. ``mcdc/object_/settings.py`` — simulation settings and k-eigenvalue parameters
#. ``mcdc/object_/material.py`` — material and composition definition
#. ``mcdc/object_/transport_model_data.py`` — particle-specific transport data
#. ``mcdc/object_/surface.py`` — surface geometry (``Surface`` class methods)
#. ``mcdc/object_/cell.py`` — cell definitions (``Cell``)
#. ``mcdc/object_/source.py`` — source specifications (``Source``)
#. ``mcdc/object_/tally.py`` — tally objects (``Tally``)
#. ``mcdc/object_/technique.py`` — variance reduction techniques
#. ``mcdc/config.py`` — command-line argument definitions

-------
Testing
-------

See :doc:`continuous_integration` for more information on how we run these tests automatically.

MC/DC has a robust testing suite that your changes must be able to pass before a PR is accepted.
Unit tests for functions that have them are ran in a pure python from.
Mostly this is for ensuring input operability
A regression test suite (including models with analytical and experimental solutions) is provided to ensure accuracy and precision of MC/DC.

Our test suite runs on every PR, and Push.
Our github based CI runs for, 

* linux-64 (x86)
* osx-64 (x86, intel based macs)

while we do not have continuous integration we have validated MC/DC on other systems.

To run the default fast unit-test suite locally, run,

.. code-block:: sh

    python -m pytest

To run the full unit-test suite in both Python and Numba mode, run,

.. code-block:: sh

    python -m pytest test/unit

To run the regression tests locally, run,

.. code-block:: sh

    python -m pytest test/regression <OPTION_FLAG(s)>


and all the tests will run. Various option ``OPTION_FLAG`` are accepted to control the tests ran,

* Run a specific test (with wildcard ``*`` support): ``--name=<test_name>``
* Skip a specific test (with wildcard ``*`` support): ``--skip=<test_name>``
* Run in Numba mode: ``--mode=numba``
* Run against the GPU target: ``--target=gpu``
* Run in multiple MPI ranks (currently support ``mpiexec`` and ``srun``): ``--mpiexec=<number of ranks>``
* Run with Slurm ``srun`` instead of ``mpiexec``: ``--srun=<number of ranks>``

Note that flags can be combined. To add a new test:

#. Create a folder. The name of the folder will be the test name.
#. Add the input file. Name it`input.py`.
#. Add the answer key file. Name it `answer.h5`.
#. Make sure that the number of particles run is large enough for a good test.
#. If the test runs longer than 5 seconds, consider decreasing the number of particles.

When adding a new hardware backend a new instantiation of the test suit should be made.
This is done with github actions. 
See the (``.github/workflows``) for examples.

If a new simulation type is added (e.g. quasi montecarlo w/ davidson's method, residual monte carlo, intrusive uq) more regression tests should be added with your PR.
If you are wondering accommodations.

--------------------
Adding Documentation
--------------------

Documentation is a core part of MC/DC.
Contributions that introduce new features, modify existing behavior, or change developer workflows should update the relevant documentation accordingly.

See the :doc:`../developer_guide/documentation/index` guide for documentation philosophy, writing guidelines, and instructions for contributing to the documentation.
