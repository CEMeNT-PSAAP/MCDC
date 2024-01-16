---
title: 'Monte Carlo / Dynamic Code (MC/DC): An accelerated Python package for fully transient neutron transport and rapid methods development'
tags:
  - Python
  - Monte Carlo
  - nuclear engineering
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
Knowing this is required for both the design and safety anylisys of these nuclear systems.
The movement of these neutrons through phase space (3 position, 2 direction of travel, 1 particle speed, and 1 time) can be modeled with a Monte Carlo simulation.
Previous Monte Carlo neutron transport applications have been built using compiled languages and predominantly conduct steady state anylisys.
When moving to dynamic simulations novel numerical methods are required to compute a solution performantly.
We designed Monte Carlo / Dynamic Code (MC/DC) to explore these novel numerical methods on modern high performance compute systems.
We avoid the need of a compiled or domain specific language by use of the Numba compiler for Python to accelerate and abstract our compute kernels to near compiled code speeds.
Using this scheme we have implemented many novel algorithms and in some verification tests we approach the performance of nominally developed industry standard codes.


# Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Future Work

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

This work was supported by the Center for Exascale Monte-Carlo Neutron Transport (CEMeNT) a PSAAP-III project funded by the Department of Energy, grant number: DE-NA003967.

# References