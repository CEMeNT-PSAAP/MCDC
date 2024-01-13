.. MC/DC documentation master file, created by
   sphinx-quickstart on Fri Oct 27 14:14:47 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


=================================
MC/DC: Monte Carlo Dynamic Code
=================================

MC/DC is a performant, scalable, and machine-portable Python-based 
Monte Carlo neutron transport software in active development 
by the `Center for Exascale Monte Carlo Neutron Transport <https://cement-psaap.github.io/>`_ (CEMeNT).

.. note::


   The project is in the early stages of *very* active development,
   not even an alpha release! 


MC/DC is machine portable and is validated to run on:

* linux-64 (x86)
* osx-64 (x86, intel based macs)
* osx-arm64 (apple silicon based macs)
* linux-ppc64 (IBM POWER9)
* linux-nvidia-cuda
* linux-amd-gcn
* win-64 (runs but not recommend!)



.. only:: html

   --------
   Contents
   --------

.. toctree::
   :maxdepth: 1

   install
   contribution
   pythonapi/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


To build the docs
=================

#. Install dependencies (we recommend: ``conda install sphinx`` and ``pip install furo``). Note that these dependencies are not installed as a part of base MC/DC
#. Run ``make html`` to compile
#. Then launch ``build/html/index.html`` with your browser of choice