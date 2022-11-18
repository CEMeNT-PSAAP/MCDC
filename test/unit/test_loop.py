import numpy as np
from mcdc.loop import loop_particle, loop_main, loop_source
import mcdc.type_ as type_
import mcdc.card
from mcdc.constant import INF



def loop_setup_test():
    type_.make_type_surface(1)
    P = np.zeros(1, dtype=type_.particle)[0]
    S = np.zeros(1, dtype=type_.surface)[0]
    
    x     = 5.0
    trans = np.array([0.0, 0.0, 0.0])

    S['G']      = 1.0
    S['linear'] = True
    S['J']      = np.array([-x, -x])
    S['t']      = np.array([0.0, INF])

    # Surface on the left
    P['x'] = 4.0
    
    # setup_card = card.InputCard()
    

    #mcdc = 

    return(S, P, mcdc)


def test_loop_main():
    
    #[P, S] = loop_setup_test()




    print('This worked')
    assert 0 == 0



def test_loop_particle():
    print('This worked')
    assert 0 == 0
    
