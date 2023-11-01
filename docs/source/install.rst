.. _install:

===================
Installation Guide
===================

This outlines the basic steps to install MC/DC on a local
machine or on HPC machine.

-----------------------------------
Creating an MC/DC Conda environment
-----------------------------------

`Conda <https://conda.io/en/latest/>`_ is an open source package and environment management system 
that runs on Windows, macOS, and Linux. It allows for easy installing and switching between multiple
versions of software packages and their dependendencies. 
We can't force you to use it, but we do *highly* recommend it, particularly
if you plan on running MC/DC in `numba mode <https://numba.pydata.org/>`_.
**The included installation script will fail if executed outside of a conda environment.**

First, `conda` should be installed with `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
or `Anaconda <https://www.anaconda.com/>`_. HPC instructions: 

`Quartz <https://hpc.llnl.gov/hardware/compute-platforms/quartz>`_ (LLNL, x86_64), 

.. code-block:: sh

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh


`Lassen <https://hpc.llnl.gov/hardware/compute-platforms/lassen>`_ (LLNL, IBM Power9),

.. code-block:: sh

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh
    bash Miniconda3-latest-Linux-ppc64le.sh


Then create and activate a new conda environment called `mcdc-env` in
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


