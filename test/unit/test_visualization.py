import mcdc.visualizer
import numpy as np

def generate_test_input():

    #testing ability to make a box
    m1 = m = mcdc.material(capture=np.array([0.05]), scatter=np.array([[0.05]]), name = "test material 1")
    
    plane_x1 = mcdc.surface("plane-x", x=0.0, bc="reflective")
    plane_x2 = mcdc.surface("plane-x", x=10.0)
    plane_y1 = mcdc.surface("plane-y", y=0.0, bc="reflective")
    plane_y2 = mcdc.surface("plane-y", y=10.0)
    plane_z1 = mcdc.surface("plane-z", z=0.0, bc="reflective")
    plane_z2 = mcdc.surface("plane-z", z=10.0)

    mcdc.cell([+plane_x1, -plane_x2, +plane_y1, -plane_y2, +plane_z1, -plane_z2], m1)

    #testing ability to make a geometry with a sphere and plane
    m2 = mcdc.material(capture=np.array([1E5]), speed=np.array([1E3]), name = "test material 2")
    
    sphere = mcdc.surface('sphere', center=[-2.0,0.0,0.0], radius=6.0)
    plane_x3 = mcdc.surface('plane-x', x=-3.5)

    
    mcdc.cell([sphere, -plane_x3], m2)

    #Testing ability to make a cylender
    m3 = mcdc.material(capture=np.array([5e-5]), scatter=np.array([[5e-5]]), speed=np.array([1E3]), name = "test material 3")

    cylinder = mcdc.surface("cylinder-x", center=[0.0, 0.0], radius=0.5)
    plane_x4 = mcdc.surface('plane-x', x=[-22.0, 22.0-12.0], t=[0.0, 5.0])
    plane_x5 = mcdc.surface('plane-x', x=[-22.0+12.0, 22.0], t=[0.0, 5.0])

    mcdc.cell([-cylinder, +plane_x4, -plane_x5], m3)

def test_material_colors():
    color_key_list = mcdc.visualize.draw_geometry(0)

    #test color key
    assert color_key_list[0][1] == "test material 1"
    assert color_key_list[1][1] == "test material 2"
    assert color_key_list[2][1] == "test material 3"
    
    #ensure colors are visually different
    assert color_key_list [0][0] != color_key_list[1][0]
    assert color_key_list [1][0] != color_key_list[2][0]
    assert color_key_list [2][0] != color_key_list[0][0]