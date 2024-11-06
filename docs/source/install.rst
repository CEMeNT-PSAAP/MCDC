.. _install:

===================
Installation Guide
===================

Developers in MC/DC (on any machine) or users on HPC machines should install using the installation script included with the source code; 
start by :ref:`creating-a-conda-environment`. 
Installing from source via the installation script is the most resilient way to get properly configured dependencies.
Most other users can install using pip. 

-------------------
Installing with pip
-------------------
Assuming you have a working Python environment and do not need to develope in MC/DC can install using pip. 
We recommend doing so within an active venv or conda (or other environment manager) environment, 
which avoids the need for any admin access and keeps dependencies clean. 

.. code-block:: sh

    pip install mcdc


-------------------------
Installing MC/DC via venv
-------------------------

Python `virtual environments <https://docs.python.org/3.11/library/venv.html>`_ are the easiest and recommended way to get MC/DC operating on personal machines as well as HPCs.
Usually this can be done by

.. code-block:: sh

    python -m venv <name_of_venv>

This assumes you have a working Python version with appropriate venv packages installed on your system.
Often HPC's have these in a module system.
Use module load ``python/<version_number>`` then launch this command.
This will give you a non-admin way of install Python projects like MC/DC and its pre-recs.
Once you have created a venv activate it with

.. code-block:: sh

    source <name_of_venv>/bin/activate

Activating the venv will need to be done every time a new terminal instance is launched.
from there MC/DC can be installed from PyPI (``pip install mcdc``) or

Installing from source (venv)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install MC/DC from source in a Python virtual environment 

#. clone MC/DC repo: ``git clone https://github.com/CEMeNT-PSAAP/MCDC.git``
#. Go to MC/DC directory ``cd MC/DC``
#. Install as an editable file ``pip install -e .``

which should install all needed dependencies without a hitch.

.. _creating-a-conda-environment:

--------------------------
Installing MC/DC via conda
--------------------------

Conda is the most robust (works even on bespoke systems) option to install MC/DC.
`Conda <https://conda.io/en/latest/>`_ is an open source package and environment management system 
that runs on Windows, macOS, and Linux. It allows for easy installing and switching between multiple
versions of software packages and their dependencies. 
Conda is really useful on systems with non-standard hardware (e.g. not x86 CPUs) like Lassen.
mpi4py is often the most troublesome dependency 

First, ``conda`` should be installed with `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
or `Anaconda <https://www.anaconda.com/>`_. HPC instructions: 

`Dane <https://hpc.llnl.gov/hardware/compute-platforms/dane>`_ (LLNL, x86_64), 

.. code-block:: sh

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh


`Lassen <https://hpc.llnl.gov/hardware/compute-platforms/lassen>`_ (LLNL, IBM Power9),

.. code-block:: sh

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh
    bash Miniconda3-latest-Linux-ppc64le.sh


Then create and activate a new conda environment called *mcdc-env* in
which to install MC/DC. This creates an environment with python3.11 
installed.

.. code-block:: sh

    conda create -n mcdc-env python=3.12
    conda activate mcdc-env

Installing from Source (conda)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MC/DC can be installed by entering the following commands in a terminal:

.. code-block:: sh

    git clone https://github.com/CEMeNT-PSAAP/MCDC.git
    cd MCDC


The MC/DC repository includes the script ``install.sh``, which will 
build MC/DC and all of its dependencies and execute any necessary patches.
This has been tested on Quartz, Dane, Tioga, Lassen, and Apple M2. 
The ``install.sh`` script **will fail outside of a conda environment**.

On HPC machines, the script will install mpi4py 
`from source <https://mpi4py.readthedocs.io/en/stable/install.html#using-distutils>`_.
This means that all appropriate modules must be loaded prior to executing.

On Quartz, the default modules are sufficient (``intel-classic`` and ``mvapich2``). 
On Lassen, ``module load gcc/8 cuda/11.8``. Then, 

.. code-block:: sh

    bash install.sh --hpc


On local machines, mpi4py will be installed using conda,

.. code-block:: sh

    bash install.sh 

To confirm that everything is properly installed, execute ``pytest`` from the MCDC directory. 

-------------------------------------
Configuring Continuous Energy Library
-------------------------------------

MC/DC has continuous energy transport capabilities.
We provide the library and easy install to members of CEMeNT and other close developers.
Due to export controls we cannot build a library and transport functionality in a single source.
If you are a member of CEMeNT you should have access to `this internal repo <https://github.com/CEMeNT-PSAAP/MCDC-Xsec>`_.
You an then either set a flag in the install script like,

.. code-block:: sh

    bash install.sh --config_cont_lib

or run the script after instillation as a stand alone operation with

.. code-block:: sh

    bash config_cont_energy.sh

Both these operations will clone the internal directory to your MCDC directory, untar the compressed folder, then set an environment variable in your bash script.
NOTE: this does assume you are using bash shell.


---------------------------------
GPU Operability (MC/DC+Harmonize)
---------------------------------

MC/DC supports most of it's Numba enabled features for GPU compilation and execution.
When targeting GPUs MC/DC uses the `Harmonize <https://github.com/CEMeNT-PSAAP/harmonize>`_ library as it's GPU runtime (the thing that actually executes MC/DC functions).
How this works gets a little involved but breifly Harmonize acts as the GPU runtime for MC/DC and has two major scheduling schemes incluing a novel and an event scheduler.
The event scheduler is similar to thoes implmented in OpenMC and Shift.
For more information on Harmonize and how we compile MC/DC with it see a `TOMACs article descibring the async scheduler <https://doi.org/10.1145/3626957>`_ or our publications in American Nuclear Society: Math and Comp Meeting in 2025.

Please file `Github issues promptly <https://github.com/CEMeNT-PSAAP/MCDC/issues>`_ when encountering configuration problems espically when on supported super copmuters (LLNL's `Tioga <https://hpc.llnl.gov/hardware/compute-platforms/tioga>`_, `El Capitan <https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems>`_, and `Lassen <https://hpc.llnl.gov/hardware/compute-platforms/lassen>`_)

Nvidia GPUs
^^^^^^^^^^^

To compile and execute MC/DC on Nvidia GPUs first ensure you have the `Harmonize prerecs <https://github.com/CEMeNT-PSAAP/harmonize/blob/main/install.sh>`_ (CUDA=11.8, Numba>=0.58.0) and a working MC/DC version >=0.10.0. Then,

#. clone the harmonize repo ``https://github.com/CEMeNT-PSAAP/harmonize.git``
#. install into proper Python env with ``pip install -e .``

Operability should now be enabled. 

AMD GPUs
^^^^^^^^

The prerequisites for AMD operability are slightly more complex.
Require a patch to Numba to allow for AMD target triple LLVM-IR.
It is recommended that this is done within a Python venv virtual environment.

To compile and execute MC/DC on AMD GPUs first ensure you have the `Harmonize prerecs <https://github.com/CEMeNT-PSAAP/harmonize/blob/main/install.sh>`_ (ROCm=6.0.0, Numba>=0.58.0) and a working MC/DC version >=0.11.0. Then,

#. Patch Numba to enable HIP (`instructions here <https://github.com/ROCm/numba-hip>`_)
#. Clone harmonize and `switch to the AMD <https://github.com/CEMeNT-PSAAP/harmonize/tree/amd_event_interop_revamp>`_ branch with ``git switch amd_event_interop_revamp`
#. Install Harmonize with ``pip install -e .`` or using `Harmonize's install script <https://github.com/CEMeNT-PSAAP/harmonize/tree/main>`_

Operability should now be enabled.