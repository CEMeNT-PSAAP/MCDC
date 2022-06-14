# MC/DC: Monte Carlo Dynamic Code

![mcdc_logo v1](https://user-images.githubusercontent.com/26186244/173467190-74d9b09a-ef7d-4f0e-8bdf-4a076de7c43c.svg)

MC/DC is a performant, scalable, and machine-portable Python-based Monte Carlo neutron transport software currently developed in the Center for Exascale Monte Carlo Neutron Transport ([CEMeNT](https://cement-psaap.github.io/)).

## Installation

In the root directory:

```bash
pip install -e .
```

## Usage

As an example, let us consider a simple time-dependent transport problem based on the [AZURV1 benchmark](https://inis.iaea.org/search/search.aspx?orig_q=RN:41070601):

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
mcdc.setting(N_hist=1E3)

# Run
mcdc.run()
```

If we save the input script above as `input.py`, we can run it as follows:

```bash
python input.py
```

An example to set up a multigroup (neutron energy and delayed precursor) material is provided in `MCDC/example/fixed_source/td_inf_SHEM361`.

### Numba mode

MC/DC supports transport kernel acceleration via [Numba](https://numba.readthedocs.io/en/stable/index.html)'s Just-in-Time compilation (currently only the CPU implementation). Running in Numba mode takes an *overhead* of about 15 to 80 seconds depending on the physics/features simulated; however, once compiled, the simulation runs MUCH faster than running in the Python mode.

To run in the Numba mode:

```bash
python input.py --mode=numba
```

### Running in parallel

MC/DC supports parallel simulation via [MPI4Py](https://mpi4py.readthedocs.io/en/stable/). As an example, to run on 36 processes with [SLURM](https://slurm.schedmd.com/documentation.html):

```bash
srun -n 36 python input.py
```

The MPI parallel simulation can be run both in Python and Numba mode.
