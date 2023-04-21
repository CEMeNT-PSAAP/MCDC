import numpy as np
import h5py

import mcdc


def test():
    # =========================================================================
    # Set model and run
    # =========================================================================

    n1 = mcdc.nuclide(capture=np.array([0.5]))
    n2 = mcdc.nuclide(
        capture=np.array([0.1]),
        fission=np.array([0.4]),
        nu_p=np.array([2.5]),
        sensitivity=True,
    )

    m = mcdc.material(nuclides=[(n1, 1.0), (n2, 1.0)])

    s1 = mcdc.surface("plane-x", x=-1e10, bc="reflective")
    s2 = mcdc.surface("plane-x", x=1e10, bc="reflective")

    mcdc.cell([+s1, -s2], m)

    mcdc.source(point=[0.0, 0.0, 0.0], isotropic=True)

    scores = ["flux-t"]
    mcdc.tally(
        scores=scores,
        x=np.linspace(-20.0, 20.0, 202),
        t=np.linspace(0.0, 20.0, 21),
    )

    mcdc.setting(N_particle=2e1)

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


if __name__ == "__main__":
    test()
