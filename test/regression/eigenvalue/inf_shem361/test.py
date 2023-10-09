import numpy as np
import h5py
import mcdc


def test():
    # =========================================================================
    # Set model
    # =========================================================================
    # The infinite homogenous medium is modeled with reflecting slab

    # Load material data
    with np.load("SHEM-361.npz") as data:
        SigmaC = data["SigmaC"]  # /cm
        SigmaS = data["SigmaS"]
        SigmaF = data["SigmaF"]
        nu_p = data["nu_p"]
        nu_d = data["nu_d"]
        chi_p = data["chi_p"]
        chi_d = data["chi_d"]
        G = data["G"]

    # Set material
    m = mcdc.material(
        capture=SigmaC,
        scatter=SigmaS,
        fission=SigmaF,
        nu_p=nu_p,
        chi_p=chi_p,
        nu_d=nu_d,
        chi_d=chi_d,
    )

    # Set surfaces
    s1 = mcdc.surface("plane-x", x=-1e10, bc="reflective")
    s2 = mcdc.surface("plane-x", x=1e10, bc="reflective")

    # Set cells
    c = mcdc.cell([+s1, -s2], m)

    # =========================================================================
    # Set initial source
    # =========================================================================

    source = mcdc.source(energy=np.ones(G))  # Arbitrary

    # =========================================================================
    # Set problem and tally, and then run mcdc
    # =========================================================================

    # Tally
    scores = ["flux"]
    mcdc.tally(scores=scores, g="all")

    # Setting
    mcdc.setting(N_particle=10, progress_bar=False, census_bank_buff=30)
    mcdc.eigenmode(N_inactive=1, N_active=2)
    mcdc.population_control()

    # Run
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
        assert np.allclose(a, b)

        name = "tally/" + score + "/sdev"
        a = output[name][:]
        b = answer[name][:]
        assert np.allclose(a, b)

    a = output["k_cycle"][:]
    b = answer["k_cycle"][:]
    assert np.allclose(a, b)

    a = output["k_mean"][()]
    b = answer["k_mean"][()]
    assert np.allclose(a, b)

    output.close()
    answer.close()
