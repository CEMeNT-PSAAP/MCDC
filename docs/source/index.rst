.. MC/DC documentation master file, created by
   sphinx-quickstart on Fri Oct 27 14:14:47 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


=================================
MC/DC: Monte Carlo Dynamic Code
=================================

MC/DC is a performant, scalable, and machine-portable Python-based 
Monte Carlo neutron transport software in active development.
It supports fully transient (aka dynamic) Monte Carlo transport and implements
novel methods and algorithms for neutron transport. MC/DC is purpose built to be
a rapid methods development platform for for modern HPCs and is targeting CPUs and GPUs.

MC/DC has support for continuous energy and multi-group transport.
It can solve more traditional k-eigenvalue problems (used to determine neutron population growth rates in reactors) as well as fully dynamic simulations.
It has a novel continuous geometry movement function (other codes use step functions) that allows for better modeling of things like control rods.
It also supports some simple Domain decomposition, with more complex algorithms currently being implemented.

MC/DC is machine portable and is validated to run on:

* linux-64 (x86)
* osx-64 (x86, intel based macs)
* osx-arm64 (apple silicon based macs)
* linux-ppc64 (IBM POWER9)
* linux-nvidia-cuda
* win-64 (runs but not recommend!)

Primary development is done by the `Center for Exascale Monte Carlo Neutron Transport <https://cement-psaap.github.io/>`_ (CEMeNT).

.. image:: images/home/cement-logo-1.png
   :width: 650
   :alt: cement logo
   :align: center
   :target: https://cement-psaap.github.io/

with support from the following institutions

.. image:: images/home/psaapiii.png
   :width: 200
   :alt: PSAAP-III logo
   :target: https://psaap.llnl.gov/
.. image:: images/home/DOE_logo.png
   :width: 275
   :alt: DOE logo
   :target: https://www.energy.gov/
.. image:: images/home/NNSA_Logo.png
   :width: 275
   :alt: NNSA logo
   :target: https://www.energy.gov/nnsa/national-nuclear-security-administration

.. image:: images/home/osu-logo.png
   :width: 400
   :alt: Oregon State University logo
   :target: https://oregonstate.edu/
.. image:: images/home/nd-logo.png
   :width: 125
   :target: https://www.nd.edu/
   :alt: Noter Dame logo
.. image:: images/home/SU.png
   :width: 125
   :alt: Seattle university logo
   :target: https://www.seattleu.edu/
.. image:: images/home/ncsu-logo.png
   :width: 125
   :alt: North Carolina state university logo
   :target: https://www.ncsu.edu/

Work within MC/DC has resulted in a large number of journal and conference publications, presentations.
A full exhaustive list of publications can be found on the `CEMeNT site <https://cement-psaap.github.io/publications/>`_

.. only:: html

   --------
   Contents
   --------

.. toctree::
    :maxdepth: 1

    install
    user
    contribution
    pythonapi/index

.. sidebar-links::
    :caption: Links
    :github: 
    :pypi: mcdc
    CEMeNT <https://cement-psaap.github.io>
    license <https://github.com/CEMeNT-PSAAP/MCDC/blob/main/LICENSE>

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

To Cite MC/DC
=============

If you use or devlope in MC/DC and would like to publish your results
please cite our article in the Journal of Open Source software

.. code-block:: 

    @article{morgan2024mcdc,
        booktitle = {}
        title = {},
        author = {},
        date = {},
        doi = {},
    }