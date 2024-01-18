# MC/DC: Monte Carlo Dynamic Code

![mcdc_logo v1](https://user-images.githubusercontent.com/26186244/173467190-74d9b09a-ef7d-4f0e-8bdf-4a076de7c43c.svg)

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-contributor%20covenant-green.svg)](https://www.contributor-covenant.org/version/2/1/code_of_conduct/code_of_conduct.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

MC/DC is a performant, scalable, and machine-portable Python-based Monte Carlo 
neutron transport software currently developed in the Center for Exascale Monte 
Carlo Neutron Transport ([CEMeNT](https://cement-psaap.github.io/)).

## Installation

For most users working on a single machine that they are administrators of MC/DC can be installed with

```bash
pip install mcdc
```

However when developing in MC/DC or when running on a high performance compute cluster we recommend you follow our [install guide](https://cement-psaapgithubio.readthedocs.io/en/latest/install.html).
MC/DC is MPI enabled and can use lots of different compilers which can get tricky to manage on an HPC.
MPI specifically can specific versions for specific machines the user should build `mpi4py` off of.
Our install scripts take care of this for you

## Documentation

We have documentation available at [https://cement-psaapgithubio.readthedocs.io/en/latest/](https://cement-psaapgithubio.readthedocs.io/en/latest/)
but you can build the docs yourself by following the [`README.md`]() in the /docs folder.

## Running

MC/DC has many different modes you can run in. 
A generous flexible pure python environment or a `jit` compiled fast, but restrictive, mode.
Depending on your use case you will most likely favor one over the other.

### Pure Python

To run a hypothetical input deck (for example this [slab wall problem]()) in pure python mode run:

```bash
python input.py
```

Simulation outputs will be placed in the same directory the `input.py` file is located in.

### Numba mode

MC/DC supports transport kernel acceleration via 
[Numba](https://numba.readthedocs.io/en/stable/index.html)'s Just-in-Time 
compilation (currently only the CPU implementation). Running in Numba mode takes 
an *overhead* of about 15 to 80 seconds depending on the physics/features 
simulated; however, once compiled, the simulation runs **MUCH** faster than the 
Python mode.

To run in the Numba mode:

```bash
python input.py --mode=numba
```

### Running in parallel

MC/DC supports parallel simulation via 
[MPI4Py](https://mpi4py.readthedocs.io/en/stable/). As an example, to run on 36 
processes in Numba mode with [SLURM](https://slurm.schedmd.com/documentation.html):

```bash
srun -n 36 python input.py --mode=numba
```

For system that do no use SLURM (i.e. a local system) try `mpiexec` or `mpirun` in its stead.

## Contributions

We welcome any contributions to this code base. Please keep in mind our [code of conduct]() that we do take seriously.
We work off a forking development structure where you fork this repo, make contributions then open a pull request. That code will then be reviewed by the primary developers. For more information on how to do this see our [Contribution guide]()

## Bugs and Issues

Our documentation is in the early stages of development so bare with us while we bring that upto snuff.
If you do feel that you have found a novel bug or we should be aware of feel free to [open an issue](https://github.com/CEMeNT-PSAAP/MCDC/issues)
**We are not your HPC's admins. We can only do so much*

## Testing

MC/DC uses CI to run it's unit and regression test suite. 
MC/DC also includes a verification and performance test that get ran on nightly builds on internal systems.
For specifics on how to run these tests locally [go here](https://github.com/CEMeNT-PSAAP/MCDC/tree/main/test/regression).

## Cite

## License

MC/DC is licensed under a BSD-3 clause license. We believe in open source software