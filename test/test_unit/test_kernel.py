
import numpy as np
from mcdc import type_
from mcdc.kernel import rng, rng_skip_ahead_
import mcdc.global_ as mcdc_
input_card = mcdc_.input_card
mcdc       = mcdc_.global_


def test_rn_basic():
    """
    Basic test routine for random number generator.
    Get the first 5 random numbers, skip a few, get 5 more and compare to 
    reference data from [1]
    
    Seed numbers for index 1-5, 123456-123460
    
    [1] F. B. Brown, “Random number generation with arbitrary strides”, 
        Trans. Am. Nucl. Soc, 71, 202 (1994)
    
    """
    ref_data = np.array((3512401965023503517, 5461769869401032777, 
                         1468184805722937541, 5160872062372652241, 
                         6637647758174943277, 794206257475890433,
                         4662153896835267997, 6075201270501039433,  
                         889694366662031813,  7299299962545529297))
    
    # PRNG parameters from [1]
    g       = 3512401965023503517
    c       = 0
    seed0   = 1
    mod     = 2**63

    # Arbitrary constants for dummy mcdc container
    G               = 1
    J               = 1
    Nmax_slice      = 1
    Nmax_surface    = 1
    Nmax_cell       = 1
    input_card.materials.append({'G':1, 'J':1}) # need a psuedo material for mcdc container

    # Make types for dummy mcdc container
    type_.make_type_material(G,J)
    type_.make_type_surface(Nmax_slice)
    type_.make_type_cell(Nmax_surface)
    type_.make_type_universe(Nmax_cell)
    type_.make_type_lattice(input_card.lattices)
    type_.make_type_source(G)
    type_.make_type_tally(input_card)
    type_.make_type_technique(input_card)
    type_.make_type_global(input_card)

    # The dummy container
    mcdc = np.zeros(1, dtype=type_.global_)[0]
    
    # Change mcdc PRNG parameters
    mcdc['setting']['rng_g']    = g
    mcdc['setting']['rng_c']    = c
    mcdc['setting']['rng_mod']  = mod
    mcdc['rng_seed_base']       = seed0
    mcdc['rng_seed']            = seed0

    # run through the first five seeds (1-5)
    for i in range(5):
        seed = (rng(mcdc)*mod)
        assert( seed == ref_data[i])
    
    # skip to 123456-123460
    rng_skip_ahead_(123456, mcdc)
    seed = mcdc['rng_seed']
    for i in range(5,10):
        assert (seed == ref_data[i])
        seed = (rng(mcdc)*mod)


    