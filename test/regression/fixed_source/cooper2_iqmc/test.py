import numpy as np
import h5py

import mcdc


def test():
    # =============================================================================
    # Set model
    # =============================================================================
    # A shielding problem based on Problem 2 of [Coper NSE 2001]
    # https://ans.tandfonline.com/action/showCitFormats?doi=10.13182/NSE00-34

    # Set materials
    SigmaT = 5.0
    c = 0.8
    m_barrier = mcdc.material(
        capture=np.array([SigmaT]), scatter=np.array([[SigmaT * c]])
    )
    SigmaT = 1.0
    m_room = mcdc.material(capture=np.array([SigmaT]), scatter=np.array([[SigmaT * c]]))

    # Set surfaces
    sx1 = mcdc.surface("plane-x", x=0.0, bc="reflective")
    sx2 = mcdc.surface("plane-x", x=2.0)
    sx3 = mcdc.surface("plane-x", x=2.4)
    sx4 = mcdc.surface("plane-x", x=4.0, bc="vacuum")
    sy1 = mcdc.surface("plane-y", y=0.0, bc="reflective")
    sy2 = mcdc.surface("plane-y", y=2.0)
    sy3 = mcdc.surface("plane-y", y=4.0, bc="vacuum")

    # Set cells
    mcdc.cell([+sx1, -sx2, +sy1, -sy2], m_room)
    mcdc.cell([+sx1, -sx4, +sy2, -sy3], m_room)
    mcdc.cell([+sx3, -sx4, +sy1, -sy2], m_room)
    mcdc.cell([+sx2, -sx3, +sy1, -sy2], m_barrier)

    # =============================================================================
    # iQMC Parameters
    # =============================================================================
    N = 2e1
    Nx = Ny = 20
    maxit = 10
    tol = 1e-3
    x = np.linspace(0, 4, num=Nx + 1)
    y = np.linspace(0, 4, num=Ny + 1)
    generator = "halton"

    # fixed source in lower left corner
    fixed_source = np.zeros((Nx, Ny))
    fixed_source[0 : int(0.25 * Nx), 0 : int(0.25 * Nx)] = 1

    phi0 = np.ones((Nx, Ny))

    mcdc.iQMC(
        x=x,
        y=y,
        fixed_source=fixed_source,
        phi0=phi0,
        maxitt=maxit,
        tol=tol,
        generator=generator,
    )

    # =============================================================================
    # Set tally, setting, and run mcdc
    # =============================================================================
    # Setting
    mcdc.setting(N_particle=N, progress_bar=False)
    # Run
    mcdc.run()
    # =========================================================================
    # Check output
    # =========================================================================

    output = h5py.File("output.h5", "r")
    answer = h5py.File("answer.h5", "r")
    a = answer["tally/iqmc_flux"][:]
    b = output["tally/iqmc_flux"][:]
    output.close()
    answer.close()

    assert np.allclose(a, b)
