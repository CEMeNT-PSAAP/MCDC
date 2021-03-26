from context import mcdc
import numpy as np


# =============================================================================
# Surfaces
# =============================================================================

S = mcdc.SurfacePlaneX(4.0)
def test_SurfacePlaneX_evaluate():
    ans = [S.evaluate(mcdc.Point(1.0,2.0,3.0)),
           S.evaluate(mcdc.Point(10.0,2.0,3.0))]
    ans = np.sign(ans)
    sol = np.array([-1.0, 1.0])
    assert ans.all() == sol.all()

def test_SurfacePlaneX_distance():
    pos1, dir1 = mcdc.Point(1.0,2.0,3.0), mcdc.Point(1.0,0.0,0.0)
    pos2, dir2 = mcdc.Point(1.0,2.0,3.0), mcdc.Point(-1.0,0.0,0.0)
    pos3, dir3 = mcdc.Point(10.0,2.0,3.0), mcdc.Point(1.0,0.0,0.0)
    pos4, dir4 = mcdc.Point(10.0,2.0,3.0), mcdc.Point(-1.0,0.0,0.0)
    ans1 = S.distance(pos1, dir1)
    ans2 = S.distance(pos2, dir2)
    ans3 = S.distance(pos3, dir3)
    ans4 = S.distance(pos4, dir4)
    ans  = np.array([ans1, ans2, ans3, ans4])
    sol  = np.array([3.0, np.inf, np.inf, 6.0])
    assert ans.all() == sol.all()

# =============================================================================
# RNG-LCG
# =============================================================================

# Solutions
sol    = []
stride = 10
rng = mcdc.random.RandomLCG(stride=stride)
for i in range(50):
    sol.append(rng())

def test_RandomLCG_skip_ahead():
    rng = mcdc.random.RandomLCG(stride=stride,skip=1)
    ans = []
    for i in range(7):
        ans.append(rng())
    assert ans == sol[stride:stride+7]

def test_RandomLCG_rebase():
    rng.skip_ahead(3, change_base=True)
    rng.skip_ahead(1, change_base=True)
    rng.skip_ahead(0, change_base=True)
    ans = []
    for i in range(10):
        ans.append(rng())
    assert ans == sol[4*stride:]
