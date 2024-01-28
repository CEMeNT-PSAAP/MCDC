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
Users who:

#. are unix based (macOS, linux, etc.),
#. have a working version of openMPI (from conda, brew, or apt),
#. are using an environment manager like conda or have administrator privileges, and
#. plan to *use* MC/DC, not develop features for MC/DC

can install using pip. 
We recommend doing so within an active conda (or other environment manager) environment, 
which avoids the need for any admin access and keeps dependencies clean. 

.. code-block:: sh

    pip install mcdc

Now you're ready to run in pure Python mode!

.. _creating-a-conda-environment:

-----------------------------------
Creating an MC/DC Conda environment
-----------------------------------

`Conda <https://conda.io/en/latest/>`_ is an open source package and environment management system 
that runs on Windows, macOS, and Linux. It allows for easy installing and switching between multiple
versions of software packages and their dependencies. 
We can't force you to use it, but we do *highly* recommend it, particularly
if you plan on running MC/DC in `numba mode <https://numba.pydata.org/>`_.
**The included installation script will fail if executed outside of a conda environment.**

First, ``conda`` should be installed with `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
or `Anaconda <https://www.anaconda.com/>`_. HPC instructions: 

`Quartz <https://hpc.llnl.gov/hardware/compute-platforms/quartz>`_ (LLNL, x86_64), 

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

    conda create -n mcdc-env python=3.11
    conda activate mcdc-env

-------------------------------------------
Installing from Source on Linux or Mac OS X
-------------------------------------------

All MC/DC source code is hosted on `Github <https://github.com/CEMeNT-PSAAP/MCDC>`_.
If you have `git <https://git-scm.com>`_, you can download MC/DC by entering the
following commands in a terminal:

.. code-block:: sh

    git clone https://github.com/CEMeNT-PSAAP/MCDC.git
    cd MCDC


The MC/DC repository includes the script ``install.sh``, which will 
build MC/DC and all of its dependencies and execute any necessary patches.
This has been tested on Quartz, Lassen, and Apple M2 (as of 11/01/2023). 
The ``install.sh`` script **will fail outside of a conda environment**.

On HPC machines, the script will install mpi4py 
`from source <https://mpi4py.readthedocs.io/en/stable/install.html#using-distutils>`_.
This means that all appropriate modules must be loaded prior to executing.

On Quartz, the default modules are sufficient (``intel-classic`` and ``mvapich2``). 
On Lassen, ``module load gcc/8 cuda/11.3``. Then, 

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
