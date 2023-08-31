import numpy as np
from mcdc import type_
from mcdc.kernel import rng
import mcdc.global_ as mcdc_

input_deck = mcdc_.input_deck


def test_rn_basic():
    """
    Basic test routine for random number generator.
    Get the first 5 random numbers, skip a few, get 5 more and compare to
    reference data from [1]

    Seed numbers for index 1-5, 123456-123460

    [1] F. B. Brown, “Random number generation with arbitrary strides”,
        Trans. Am. Nucl. Soc, 71, 202 (1994)

    """
    ref_data = np.array(
        (
            1,
            2806196910506780710,
            6924308458965941631,
            7093833571386932060,
            4133560638274335821,
        )
    )

    # PRNG parameters from [1]
    g = 2806196910506780709
    c = 1
    seed0 = 1
    mod = 2**63

    # Arbitrary constants for dummy mcdc container
    G = 1
    J = 1
    Nmax_slice = 1
    Nmax_surface = 1
    Nmax_cell = 1
    # need a psuedo material for mcdc container
    input_deck.materials.append({"G": 1, "J": 1})

    # Make types for dummy mcdc container
    type_.make_type_material(G, J, 1)
    type_.make_type_surface(Nmax_slice)
    type_.make_type_cell(Nmax_surface)
    type_.make_type_universe(Nmax_cell)
    type_.make_type_lattice(input_deck.lattices)
    type_.make_type_source(G)
    type_.make_type_tally(1, input_deck.tally)
    type_.make_type_technique(0, 1, input_deck.technique)
    type_.make_type_global(input_deck)

    # The dummy container
    mcdc = np.zeros(1, dtype=type_.global_)[0]

    # run through the first five seeds (1-5)
    for i in range(5):
        rng(mcdc['setting'])
        assert mcdc['setting']["rng_seed"] == ref_data[i]
