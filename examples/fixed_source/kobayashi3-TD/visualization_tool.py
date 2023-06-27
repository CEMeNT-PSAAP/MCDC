#MC/DC imports
import numpy as np
import mcdc

#Visualization imports
import netgen.gui #launches visualiztation window

from ngsolve import Draw, Redraw # just for visualization

from netgen.meshing import *
from netgen.csg import *


# =============================================================================
# Set model
# =============================================================================
# Based on Kobayashi dog-leg benchmark problem
# (PNE 2001, https://doi.org/10.1016/S0149-1970(01)00007-5)

m = mcdc.material(capture=np.array([0.05]), scatter=np.array([[0.05]]))
m_void = mcdc.material(capture=np.array([5e-5]), scatter=np.array([[5e-5]]))

sx1 = mcdc.surface("plane-x", x=0.0, bc="reflective")
sx2 = mcdc.surface("plane-x", x=10.0)
sx3 = mcdc.surface("plane-x", x=30.0)
sx4 = mcdc.surface("plane-x", x=40.0)
sx5 = mcdc.surface("plane-x", x=60.0, bc="vacuum")
sy1 = mcdc.surface("plane-y", y=0.0, bc="reflective")
sy2 = mcdc.surface("plane-y", y=10.0)
sy3 = mcdc.surface("plane-y", y=50.0)
sy4 = mcdc.surface("plane-y", y=60.0)
sy5 = mcdc.surface("plane-y", y=100.0, bc="vacuum")
sz1 = mcdc.surface("plane-z", z=0.0, bc="reflective")
sz2 = mcdc.surface("plane-z", z=10.0)
sz3 = mcdc.surface("plane-z", z=30.0)
sz4 = mcdc.surface("plane-z", z=40.0)
sz5 = mcdc.surface("plane-z", z=60.0, bc="vacuum")

surface_list = [sx1 , sx2, sx3, sx4, sx5, sy1, sy2, sy3, sy4, sy5, sz1, sz2, sz3, sz4, sz5] #a list with all the surfaces

cell1 = mcdc.cell([+sx1, -sx2, +sy1, -sy2, +sz1, -sz2], m)
# Voids
cell2 = mcdc.cell([+sx1, -sx2, +sy2, -sy3, +sz1, -sz2], m_void)
cell3 = mcdc.cell([+sx1, -sx3, +sy3, -sy4, +sz1, -sz2], m_void)
cell4 = mcdc.cell([+sx3, -sx4, +sy3, -sy4, +sz1, -sz3], m_void)
cell5 = mcdc.cell([+sx3, -sx4, +sy3, -sy5, +sz3, -sz4], m_void)
# Shield
cell6 = mcdc.cell([+sx1, -sx3, +sy1, -sy5, +sz2, -sz5], m)
cell7 = mcdc.cell([+sx2, -sx5, +sy1, -sy3, +sz1, -sz2], m)
cell8 = mcdc.cell([+sx3, -sx5, +sy1, -sy3, +sz2, -sz5], m)
cell9 = mcdc.cell([+sx3, -sx5, +sy4, -sy5, +sz1, -sz3], m)
cell10 = mcdc.cell([+sx4, -sx5, +sy4, -sy5, +sz3, -sz5], m)
cell11 = mcdc.cell([+sx4, -sx5, +sy3, -sy4, +sz1, -sz5], m)
cell12 = mcdc.cell([+sx3, -sx4, +sy3, -sy5, +sz4, -sz5], m)
cell13 = mcdc.cell([+sx1, -sx3, +sy4, -sy5, +sz1, -sz2], m)

cell_list = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9, cell10, cell11, cell12, cell13] # list with all the cells



geo = CSGeometry() #create the geometry object


for cell in cell_list:
    cube_side_list = []
    for i in range(0,len(cell["surface_IDs"])):
        surface_ID = cell["surface_IDs"][i]
        if (surface_list[surface_ID].type == "plane-x"):
            point = (abs(surface_list[surface_ID].card["J"][0][0]),0,0)
            if (not(cell["positive_flags"][i])):
                vector = (surface_list[surface_ID].card["G"], surface_list[surface_ID].card["H"], surface_list[surface_ID].card["I"])
            else:
                vector = (-surface_list[surface_ID].card["G"], surface_list[surface_ID].card["H"], surface_list[surface_ID].card["I"])
            cube_side_list.append(Plane(Pnt(point), Vec(vector)).col([1,0,0]).transp())
        elif(surface_list[surface_ID].type == "plane-y"):
            point = (0, abs(surface_list[surface_ID].card["J"][0][0]),0)
            if (not(cell["positive_flags"][i])):
                vector = (surface_list[surface_ID].card["G"], surface_list[surface_ID].card["H"], surface_list[surface_ID].card["I"])
            else:
                vector = (surface_list[surface_ID].card["G"], -surface_list[surface_ID].card["H"], surface_list[surface_ID].card["I"])
            cube_side_list.append(Plane(Pnt(point), Vec(vector)).col([1,0,0]).transp())

        elif(surface_list[surface_ID].type == "plane-z"):
            point = (0,0,abs(surface_list[surface_ID].card["J"][0][0]))
            if (not(cell["positive_flags"][i])):
                vector = (surface_list[surface_ID].card["G"], surface_list[surface_ID].card["H"], surface_list[surface_ID].card["I"])
            else:
                vector = (surface_list[surface_ID].card["G"], surface_list[surface_ID].card["H"], -surface_list[surface_ID].card["I"])
            cube_side_list.append(Plane(Pnt(point), Vec(vector)).col([1,0,0]).transp())
    
    # Set the sides of each box
    left = cube_side_list[0]
    right = cube_side_list[1]
    front = cube_side_list[2]
    back = cube_side_list[3]
    top = cube_side_list[4]
    bot = cube_side_list[5]

    #make the cube by intersecting the six planes
    cube  = left * right * front * back * bot * top

    #add cubes to the geometry by material. Assign each material its own color
    if cell["material_ID"] == 1:
        print(0)

        #geo.Add(cube.col([1,0,0]))
        
    else:
        print(0)
        #Solid
        geo.Add((cube.col([0,1,0]))) 
        #Transparent 
        #geo.Add((cube.col([0,1,0])), transparent = True)
        
       
geo.Draw()
Redraw()

