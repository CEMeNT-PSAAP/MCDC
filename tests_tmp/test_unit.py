import numpy as np

import sys

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../')

import mcdc


# =============================================================================
# TODO: Point, Particle, ...
# =============================================================================

# =============================================================================
# Surfaces
# =============================================================================

def test_SurfacePlaneX_evaluate():
    S = mcdc.SurfacePlaneX(4.0)
    ans = [S.evaluate(mcdc.Point(1.0,2.0,3.0)),
           S.evaluate(mcdc.Point(10.0,2.0,3.0))]
    ans = np.sign(ans)
    sol = np.array([-1.0, 1.0])
    assert ans.all() == sol.all()

def test_SurfacePlaneX_distance():
    S = mcdc.SurfacePlaneX(4.0)
    unit = 1.0/(3.0**0.5)
    pos1, dir1 = mcdc.Point(1.0,2.0,3.0),  mcdc.Point(unit,unit,unit)
    pos2, dir2 = mcdc.Point(1.0,2.0,3.0),  mcdc.Point(-unit,-unit,-unit)
    pos3, dir3 = mcdc.Point(10.0,2.0,3.0), mcdc.Point(unit,unit,unit)
    pos4, dir4 = mcdc.Point(10.0,2.0,3.0), mcdc.Point(-unit,-unit,-unit)
    ans1 = S.distance(pos1, dir1)
    ans2 = S.distance(pos2, dir2)
    ans3 = S.distance(pos3, dir3)
    ans4 = S.distance(pos4, dir4)
    ans  = np.array([ans1, ans2, ans3, ans4])
    sol  = np.array([3.0/unit, np.inf, np.inf, 6.0/unit])
    assert ans.all() == sol.all()

# =============================================================================
# RNG-LCG
# =============================================================================

# Check seed sequence against Brown's 2005
#  index: 1-5, 123456-123460
def test_RandomLCG_sequence():
    sol = [3512401965023503517, 5461769869401032777, 1468184805722937541,
           5160872062372652241, 6637647758174943277,  794206257475890433,
           4662153896835267997, 6075201270501039433,  889694366662031813,
           7299299962545529297]
    ans = []
    rng = mcdc.random.RandomLCG(g=mcdc.constant.LCG_G_B05, 
                                c=mcdc.constant.LCG_C_B05)
    for i in range(5): 
        rng()
        ans.append(rng.seed)
    for i in range(5, 123455): rng()
    for i in range(5):
        rng()
        ans.append(rng.seed)
    assert ans == sol

# Skip-ahead and rebase
def test_RandomLCG_skip_ahead_rebase():
    # Solutions
    sol    = []
    stride = 10
    rng = mcdc.random.RandomLCG(stride=stride)
    for i in range(50):
        sol.append(rng())

    # Skip-ahead
    rng  = mcdc.random.RandomLCG(stride=stride,skip=1)
    ans1 = []
    for i in range(7):
        ans1.append(rng())

    # Rebase
    rng.skip_ahead(3, rebase=True)
    rng.skip_ahead(1, rebase=True)
    rng.skip_ahead(0, rebase=True)
    ans2 = []
    for i in range(10):
        ans2.append(rng())
    assert ans1 == sol[stride:stride+7] and ans2 == sol[4*stride:]
