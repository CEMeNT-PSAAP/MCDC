# MC/DC: Monte Carlo Dynamic Code

![mcdc_logo v1](https://user-images.githubusercontent.com/26186244/173467190-74d9b09a-ef7d-4f0e-8bdf-4a076de7c43c.svg)

[![Build](https://github.com/CEMeNT-PSAAP/MCDC/actions/workflows/mpi_numba_reg.yml/badge.svg)](https://github.com/CEMeNT-PSAAP/MCDC/actions/workflows/mpi_numba_reg.yml)
[![ReadTheDocs](https://readthedocs.org/projects/cement-psaapgithubio/badge/?version=latest&style=flat)](https://cement-psaapgithubio.readthedocs.org/en/latest/ )
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)



MC/DC is a performant, scalable, and machine-portable Python-based Monte Carlo 
neutron transport software currently developed in the Center for Exascale Monte 
Carlo Neutron Transport ([CEMeNT](https://cement-psaap.github.io/)).

We have documentation on, installation, contribution and a brief user guide using [Read the Docs](https://cement-psaapgithubio.readthedocs.io/en/latest/)

## Installation

We recommend using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) or some other environment manager to manage the MC/DC installation.
This avoids the need for sudo access for MC/DC's dependencies and allows greater configurability for developers.
For most users working on a single machine that they are administrators of MC/DC can be installed with:
```bash
pip install mcdc
```

### Common issues with `mpi4py`

The `pip mpi4py` distribution commonly has errors when building due to incompatible local MPI dependencies it builds off of. While pip does have some reamdie for this we recommend the following:
* **Mac users:** we recommend `openmpi` is [installed via homebrew](https://formulae.brew.sh/formula/open-mpi) (note that more reliable mpi4py distribution can also be [found on homebrew](https://formulae.brew.sh/formula/mpi4py)), alternatively you can use `conda` if you don't have admin privileges;
* **Linux users:** we recommend `openmpi` is installed via a root package manager if possible (e.g. `sudo apt install openmpi`) or a conda distribution (e.g. `conda install openmpi`)
* **HPC users and developers on any system:** we recommend you *do not use the pip distribution* and instead install MC/DC and its dependencies via [install scripts](https://cement-psaapgithubio.readthedocs.io/en/latest/install.html) we've included that will build `mpi4py` from source and use conda to manage your environment. *This is the most reliable way MC/DC can be installed and configured*. It also takes care of the [Numba patch](), and can configure the [continuous energy data library]() if you have access.

### Numba Config

MC/DC requires Numba to be patched to be able to run performantly in `numba mode`. There is a very simple patch file that will make it automatically. To patch numba
1. Download the `patch.sh` file [here]() (If you cloned the git packa)
2. Make sure you have bash installed if nessacary (macOS)
3. In your active conda environment run `bash patch_numba.sh`
*If you manage your environment with conda you will not need admin privileges*

### `visualizer` Config

MC/DC has a visualizer built from the [netgen package](https://ngsolve.org/). This is not included in the base install of MC/DC due to the size of the dependencies (~300MB). To install these dependencies use
```bash
pip install mcdc['viz']
```

## Running

MC/DC has many different modes you can run in. 
A generous flexible pure python environment or a `jit` compiled fast, but restrictive, mode.
Depending on your use case you will most likely favor one over the other.

### Pure Python

To run a hypothetical input deck (for example this [slab wall problem](https://github.com/CEMeNT-PSAAP/MCDC/tree/main/examples/fixed_source/slab_absorbium)) in pure python mode run:

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

We welcome any contributions to this code base.
Please keep in mind our [code of conduct](https://github.com/CEMeNT-PSAAP/MCDC/blob/main/CODE_OF_CONDUCT.md) that we do take seriously.
We work off a forking development structure where you fork this repo, make contributions then open a pull request. That code will then be reviewed by the primary developers. For more information on how to do this see our [Contribution guide]()

## Bugs and Issues

Our documentation is in the early stages of development so bare with us while we bring that upto snuff.
If you do feel that you have found a novel bug or we should be aware of feel free to [open an issue](https://github.com/CEMeNT-PSAAP/MCDC/issues).

## Testing

MC/DC uses CI to run it's unit and regression test suite. 
MC/DC also includes a verification and performance test that get ran on nightly builds on internal systems.
See specifics on how to run these tests locally [here](https://github.com/CEMeNT-PSAAP/MCDC/tree/main/test/regression).

## Cite

To provide proper attribution to MC/DC please cite,
```
@inproceedings{var_mc23_mcdc,
    Booktitle = {International Conference on Mathematics and Computational Methods Applied to Nuclear Science and Engineering},
    title = {Development of {MC/DC}: a performant, scalable, and portable Python-based {M}onte {C}arlo neutron transport code},
    year = {2023},
    author = {Ilham Variansyah and Joanna Piper Morgan and Kyle E. Niemeyer and Ryan G. McClarren},
    address = {Niagara Falls, Ontario, Canada},
    doi={10.48550/arXiv.2305.07636},
}
```
which should render something like this

Variansyah, Ilham, J. P. Morgan, K. E. Niemeyer, and R. G. McClarren. 2023. “Development of MC/DC: a performant, scalable, and portable Python-based Monte Carlo neutron transport code.” In *International Conference on Mathematics and Computational Methods Applied to Nuclear Science and Engineering*, Niagara Falls, Ontario, Canada. DOI. [10.48550/arXiv.2305.07636](https://doi.org/10.48550/arXiv.2305.07636)

## License

MC/DC is licensed under a BSD-3 clause license. We believe in open source software