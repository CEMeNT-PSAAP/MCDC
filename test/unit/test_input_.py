

from mcdc.input_ import surface
import mcdc.type_
import numpy as np

def test_surface_input_lower():
    type_ = 'pLaNe x'
    result = surface(type_, bc='RefLeCtiVe',x=0.0)
    assert (result.card['reflective'])
    assert (result.card['A']==0.0)
    assert (result.card['B']==0.0)
    assert (result.card['C']==0.0)
    assert (result.card['D']==0.0)
    assert (result.card['E']==0.0)
    assert (result.card['F']==0.0)
    assert (result.card['G']==1.0)
    assert (result.card['H']==0.0)
    assert (result.card['I']==0.0)
    assert (result.card['J']==np.array([[-0.,  0.]])).all()
    assert (result.card['linear'])
