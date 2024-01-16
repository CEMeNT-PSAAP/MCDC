---
title: 'Monte Carlo / Dynamic Code (MC/DC): An accelerated Python package for fully transient neutron transport and rapid methods development'
tags:
  - Python
  - Monte Carlo
  - nuclear engineering
  - neutron transport
  - reactor anylisys
  - numba
  - HPC
  - mpi4py
authors:
  - name: Joanna Piper Morgan
    orcid: 0000-0003-1379-5431
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
    corresponding: true
  - name: Ilham Variansyah 
    orcid: 0000-0003-3426-7160
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: "1, 2"
    corresponding: true
  - name: Samuel Pasmann
    orcid: 0000-0003-1391-1471
    equal-contrib: true
    affiliation: "1, 3"
  - name: Kayla Beth Clements
    orcid: 0000-0003-3358-5618
    equal-contrib: true
    affiliation: "1, 2"
  - name: Braxton Cuneo
    orcid: 0000-0002-6493-0990
    equal-contrib: true
    affiliation: "1, 5"
  - name: Alexander Mote 
    orcid: 0000-0001-5099-0223
    equal-contrib: true
    affiliation: "1, 2"
  - name: Charles Goodman
    orcid: 
    equal-contrib: true
    affiliation: "1, 4"
  - name: Caleb Shaw 
    orcid: 
    equal-contrib: true
    affiliation: "1, 4"
  - name: Jordan Northrop 
    orcid: 0000-0003-0420-9699
    affiliation: "1, 2"
  - name: Rohan Pankaj
    orcid: 
    affiliation: "1, 6"
  - name: Ethan Lame
    orcid: 0000-0001-7686-9755
    affiliation: "1, 2"
  - name:  Benjamin Whewell
    orcid: 0000-0001-7826-5525
    affiliation: "1, 3"
  - name: Ryan McClarren #advisors in order of authors except Niemeyer
    orcid: 0000-0002-8342-6132
    affiliation: "1, 3"
  - name: Todd Palmer
    affiliation: "1, 2"
  - name: Lizhong Chen 
    orcid: 0000-0001-5890-7121
    affiliation: "1, 2"
  - name: Dmitriy Anistratov
    affiliation: "1, 4"
  - name: C. T. Kelley
    affiliation: "1, 4"
  - name: Camille Palmer
    orcid: 0000-0002-7573-4215
    affiliation: "1, 2"
  - name: Kyle E. Niemeyer
    orcid: 0000-0003-4425-7097
    affiliation: "1, 2"


affiliations:
 - name: Center for Exascale Monte Carlo Neutron Transport
   index: 1
 - name: Oregon State University, Corvallis, OR, USA
   index: 2
 - name: University of Notre Dame, South Bend, IN, USA
   index: 3
 - name: North Carolina State University, Raleigh, NC, USA
   index: 4
 - name: Seattle University, Seattle, WA, USA
   index: 5
 - name: Brown University, Providence, RI, USA
   index: 6
date: 16 Jan 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

When, where, and how neutrons move within a nuclear reactor governs when, where, and how heat is released.
Knowing this is required for both the design and safety analysis of these nuclear systems.
The movement of neutrons through phase space (3 position, 2 direction of travel, 1 particle speed, and 1 time) can be modeled with a Monte Carlo simulation.
Previous Monte Carlo neutron transport applications have been built using compiled languages and predominantly conduct steady state analysis.
When moving to dynamic simulations novel numerical methods are required to compute a solution performantly.
We designed Monte Carlo / Dynamic Code (`MC/DC`) to explore these novel numerical methods on modern high performance compute systems.
We avoid the need of a compiled or domain specific language by use of the Numba compiler for Python to accelerate and abstract our compute kernels to near compiled code speeds.
Using this scheme we have implemented many novel algorithms and in some verification tests we approach the performance of nominally developed industry standard codes at the tens of thousands of processors scale.


# Statement of need

`MC/DC` is a performant rapid methods development platform for novel dynamic neutron transport algorithms on modern high performance compute systems.
It uses the Numba compiler for Python to compile compute kernels to a desired hardware target, including support for graphics processing units (GPUs) [@lam_numba_2015].
`MC/DC` uses `mpi4py` to gain distributed memory parallelism [@mpi4py_2021] and has run at the tens of thousands of processors scale [@variansyah_mc23_mcdc].
These acceleration and abstraction techniques allow for `MC/DC` developers to remain in a pure Python development environment without need to support compiled or domain specific languages. 
This has allowed `MC/DC` to grow form nothing to supporting full performant neutron transport and investigating novel transport algorithms in under two years with development mostly from relative novices.

Many of the traditionally developed neutron transport codes are export controlled (not open source and difficult to get) and notoriously difficult install, use, and develope in. `MC/DC` being an open source easily installable Python package means that it is ideal for an academic environment for both research and education by undergraduates on up. This is only further assisted by the full test suite we have developed to do unit, regression, verification and performance tests, most of which run on a CI basis.

`MC/DC` has support for continuous energy and multi-group transport. It can solve more traditional k-eigenvalue problems (used to determine neutron population growth rates in reactors) as well as fully dynamic simulations. It has a novel continuous geometry movement function (other codes use step functions) that allows for better modeling of things like control rods. It also supports some simple Domain decomposition, with more complex algorithms currently being implemented.

`MC/DC` enabled explorations into dynamic neutron transport algorithms have been successful, including, quasi-Monte Carlo techniques [@mcdc:variansyah_physor22_pct], hybrid iterative techniques for k-eigenvalue simulations [@mcdc:qmc; @mcdc:qmcabs], transient population control techniques [@mcdc:variansyah_nse22_pct], hash based random number generation, global uncertainty quantification [@mcdc:clements_mc23], residual Monte Carlo methods, and machine learning techniques for dynamic node scheduling among others.

# Future Work

The main `MC/DC` branch currently only supports CPU architectures enabled by Numba (x86-64, arm64, and ppc64) but we are rapidly extending support to GPUs.
We currently have operability on Nvidia GPUs (supported via Numba) and work is ongoing to enable compilation to AMD GPUs.
On GPUs `MC/DC` will use the `harmonize` asynchronous GPU scheduler to increase performance [@brax2023].
`harmonize` works by batching jobs during execution such that similar operations get executed at the same time, reducing the difference between parallel threads running on the GPU at the same time.

We will continue to explore novel methods for dynamic neutron transport and will keep pushing to make `MC/DC` not only a proven platform for rapid neutron transport methods exploration, but also a full fledged simulation code for academic and industrial use alike.

# Acknowledgements

This work was supported by the Center for Exascale Monte-Carlo Neutron Transport (CEMeNT) a PSAAP-III project funded by the Department of Energy, grant number: DE-NA003967.

# References