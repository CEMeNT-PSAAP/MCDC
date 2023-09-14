import numpy as np
import h5py

import mcdc


def test():
    # =========================================================================
    # Set model and run
    # =========================================================================
    # A shielding problem based on Problem 2 of [Coper NSE 2001]
    # https://ans.tandfonline.com/action/showCitFormats?doi=10.13182/NSE00-34

    SigmaT = 5.0
    c = 0.8
    m_barrier = mcdc.material(
        capture=np.array([SigmaT]), scatter=np.array([[SigmaT * c]])
    )
    SigmaT = 1.0
    m_room = mcdc.material(capture=np.array([SigmaT]), scatter=np.array([[SigmaT * c]]))

    sx1 = mcdc.surface("plane-x", x=0.0, bc="reflective")
    sx2 = mcdc.surface("plane-x", x=2.0)
    sx3 = mcdc.surface("plane-x", x=2.4)
    sx4 = mcdc.surface("plane-x", x=4.0, bc="vacuum")
    sy1 = mcdc.surface("plane-y", y=0.0, bc="reflective")
    sy2 = mcdc.surface("plane-y", y=2.0)
    sy3 = mcdc.surface("plane-y", y=4.0, bc="vacuum")

    mcdc.cell([+sx1, -sx2, +sy1, -sy2], m_room)
    mcdc.cell([+sx1, -sx4, +sy2, -sy3], m_room)
    mcdc.cell([+sx3, -sx4, +sy1, -sy2], m_room)
    mcdc.cell([+sx2, -sx3, +sy1, -sy2], m_barrier)

    mcdc.source(x=[0.0, 1.0], y=[0.0, 1.0], isotropic=True)

    scores = ["flux"]
    mcdc.tally(scores=scores, x=np.linspace(0.0, 4.0, 40), y=np.linspace(0.0, 4.0, 40))

    mcdc.setting(N_particle=1e1)
    mcdc.implicit_capture()

    mcdc.run()

    # =========================================================================
    # Check output
    # =========================================================================

    output = h5py.File("output.h5", "r")
    answer = h5py.File("answer.h5", "r")
    for score in scores:
        name = "tally/" + score + "/mean"
        a = output[name][:]
        b = answer[name][:]
        assert np.isclose(a, b).all()

        name = "tally/" + score + "/sdev"
        a = output[name][:]
        b = answer[name][:]
        assert np.isclose(a, b).all()

    output.close()
    answer.close()
