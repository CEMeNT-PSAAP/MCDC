
#from .test_utilities import single_source
#from ..loop import loop_particle, loop_source, loop_main
#import mcdc
"""
def test_rn_basic():
    
    Test routine for basic random number generator.
    Get the first 5 random numbers, skip a few, get 5 more.
    
    reference data:  seeds for case of init.seed = 1,
    seed numbers for index 1-5, 123456-123460

    
    ref_data = np.array((3512401965023503517, 5461769869401032777, 
                         1468184805722937541, 5160872062372652241, 
                         6637647758174943277, 794206257475890433,
                         4662153896835267997, 6075201270501039433,  
                         889694366662031813,  7299299962545529297))
    
    mcdc['rng_seed_base'] = 1
"""

import numpy as np

ref_data = np.array((3512401965023503517, 5461769869401032777, 
                     1468184805722937541, 5160872062372652241, 
                     6637647758174943277, 794206257475890433,
                     4662153896835267997, 6075201270501039433,  
                     889694366662031813,  7299299962545529297))
    
def rng(seed):
    g        = 3512401965023503517 # changed from whats in MCDC
    c        = 1
    mod      = 2**63
    mod_mask = int(mod - 1)
    seed     = (g*int(seed)) & mod_mask # why "+ c", this changes the results
    
    return seed # /mod

def rng_skip_ahead(n):
    seed_base = 1
    g         = 3512401965023503517
    c         = 1
    g_new     = 1
    c_new     = 0
    mod       = 2**63
    mod_mask  = int(mod - 1)
    
    n = n & mod_mask
    while n > 0:
        if n & 1:
            g_new = g_new*g       & mod_mask
            c_new = (c_new*g + c) & mod_mask # removed + c
    
        c = (g+1)*c & mod_mask
        g = g*g     & mod_mask
        n >>= 1
       
    seed = (g_new*int(seed_base) ) & mod_mask
       
    return seed 
    
seed = 1
print("Seed: ")
print(seed)
for i in range(5):
    # run through the first five seeds (1-5)
    seed = rng(seed)
    print(seed)
    assert seed == ref_data[i]

# skip to 123456-123460
seed = rng_skip_ahead(123456)
for i in range(5,10):
    print(seed)
    assert seed == ref_data[i]
    seed = rng(seed)
    
    
    
    
    
    
    
    
    
    