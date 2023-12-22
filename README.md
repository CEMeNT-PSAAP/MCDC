# MC/DC: Monte Carlo Dynamic Code

![mcdc_logo v1](https://user-images.githubusercontent.com/26186244/173467190-74d9b09a-ef7d-4f0e-8bdf-4a076de7c43c.svg)

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-contributor%20covenant-green.svg)](https://www.contributor-covenant.org/version/2/1/code_of_conduct/code_of_conduct.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

MC/DC is a performant, scalable, and machine-portable Python-based Monte Carlo 
neutron transport software currently developed in the Center for Exascale Monte 
Carlo Neutron Transport ([CEMeNT](https://cement-psaap.github.io/)).

*Please Note that this project is in the early stages of devlopment (not even an alpha).* That being said enjoy!

## Installation

In the root directory:

```bash
pip install -e .
```

## Usage

As an example, let us consider a simple time-dependent transport problem based 
on the [AZURV1 benchmark](https://inis.iaea.org/search/search.aspx?orig_q=RN:41070601):

```python
import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================

# Set materials
m = mcdc.material(capture = np.array([1.0/3.0]),
                  scatter = np.array([[1.0/3.0]]),
                  fission = np.array([1.0/3.0]),
                  nu_p    = np.array([2.3]),
                  speed   = np.array([1.0]))

# Set surfaces
s1 = mcdc.surface('plane-x', x=-1E10, bc="reflective")
s2 = mcdc.surface('plane-x', x=1E10,  bc="reflective")

# Set cells
mcdc.cell([+s1, -s2], m)

# =============================================================================
# Set source
# =============================================================================

mcdc.source(point=[0.0,0.0,0.0], isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally
mcdc.tally(scores=['flux'],
           x=np.linspace(-20.5, 20.5, 202),
           t=np.linspace(0.0, 20.0, 21))

# Setting
mcdc.setting(N_particle=1E3)

# Run
mcdc.run()
```

If we save the input script above as `input.py`, we can run it as follows:

```bash
python input.py
```

A more advanced input example that includes setting up multigroup (in energy and 
delayed precursor) materials, lattice geometry, and continuous movements of 
control rods is provided in `MCDC/example/c5g7/3d/TDX`.

### Output

MC/DC simulation results are stored in 
[HDF5 format](https://www.hdfgroup.org/solutions/hdf5/), which can be processed 
using [H5Py](https://www.h5py.org/) (default file name: `output.h5`) as follows:

```python
import h5py

with h5py.File('output.h5', 'r') as f:
    x      = f['tally/grid/x'][:]
    t      = f['tally/grid/t'][:]
    phi    = f['tally/flux/mean'][:]
    phi_sd = f['tally/flux/sdev'][:]
```

### Numba mode

MC/DC supports transport kernel acceleration via 
[Numba](https://numba.readthedocs.io/en/stable/index.html)'s Just-in-Time 
compilation (currently only the CPU implementation). Running in Numba mode takes 
an *overhead* of about 15 to 80 seconds depending on the physics/features 
simulated; however, once compiled, the simulation runs MUCH faster than the 
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
