#MC/DC imports
import numpy as np
import mcdc

#Visualization imports
import netgen.gui #launches visualiztation window

from ngsolve import Draw, Redraw # just for visualization

from netgen.meshing import *
from netgen.csg import *

import math

from tkinter import *


#This cell is copied from the input file for the godiva mockup

# =============================================================================
# Set model
# =============================================================================


# =============================================================================
# Materials
# =============================================================================

m_abs = mcdc.material(capture=np.array([1E5]), speed=np.array([1E3]))
m_void = mcdc.material(capture=np.array([5e-5]), scatter=np.array([[5e-5]]), speed=np.array([1E3]))

# =============================================================================
# Set surfaces
# =============================================================================

# For cube boundaries
cube_x0 = mcdc.surface('plane-x', x=-22.0, bc='vacuum')
cube_x1 = mcdc.surface('plane-x', x=22.0, bc='vacuum')
cube_y0 = mcdc.surface('plane-y', y=-12.0, bc='vacuum')
cube_y1 = mcdc.surface('plane-y', y=12.0, bc='vacuum')
cube_z0 = mcdc.surface('plane-z', z=-12.0, bc='vacuum')
cube_z1 = mcdc.surface('plane-z', z=12.0, bc='vacuum')

# For the 3-part hollow sphere
sp_left = mcdc.surface('sphere', center=[-2.0,0.0,0.0], radius=6.0)
sp_center = mcdc.surface('sphere', center=[0.0,0.0,0.0], radius=6.0)
sp_right = mcdc.surface('sphere', center=[2.0,0.0,0.0], radius=6.0)
pl_x0 = mcdc.surface('plane-x', x=-3.5)
pl_x1 = mcdc.surface('plane-x', x=-1.5)
pl_x2 = mcdc.surface('plane-x', x=1.5)
pl_x3 = mcdc.surface('plane-x', x=3.5)

# For the moving rod
cy = mcdc.surface("cylinder-x", center=[0.0, 0.0], radius=0.5)
pl_rod0 = mcdc.surface('plane-x', x=[-22.0, 22.0-12.0], t=[0.0, 5.0])
pl_rod1 = mcdc.surface('plane-x', x=[-22.0+12.0, 22.0], t=[0.0, 5.0])




# =============================================================================
# Set cells
# =============================================================================

# Moving rod
cell1 = mcdc.cell([-cy, +pl_rod0, -pl_rod1], m_void)

# 3-part hollow shpere
cell2 = mcdc.cell([-sp_left, -pl_x0, +cy], m_void)
cell3 = mcdc.cell([-sp_center, +pl_x1, -pl_x2, +cy], m_void)
cell4 = mcdc.cell([-sp_right, +pl_x3, +cy], m_void)

# Surrounding water
# Left of rod
cell5 = mcdc.cell([-cy, +cube_x0, -pl_rod0], m_abs)
# Right of rod
cell6 = mcdc.cell([-cy, +pl_rod1, -cube_x1], m_abs)
# The rest
cell7 = mcdc.cell([+cy, +sp_left, +cube_x0, -pl_x0, +cube_y0, -cube_y1, +cube_z0, -cube_z1], m_abs)
cell8 = mcdc.cell([+cy, +pl_x0, -pl_x1, +cube_y0, -cube_y1, +cube_z0, -cube_z1], m_abs)
cell9 = mcdc.cell([+sp_center, +pl_x1, -pl_x2, +cube_y0, -cube_y1, +cube_z0, -cube_z1], m_abs)
cell10 = mcdc.cell([+cy, +pl_x2, -pl_x3, +cube_y0, -cube_y1, +cube_z0, -cube_z1], m_abs)
cell11 = mcdc.cell([+cy, +sp_right, +pl_x3, -cube_x1, +cube_y0, -cube_y1, +cube_z0, -cube_z1], m_abs)


#cell_list = [cell1]

# =============================================================================
# Set source
# =============================================================================

source = mcdc.source(
    x=[-22.0, 22.0], time=[0.0, 5.0], isotropic=True
)

#create a cell and surface list
cell_list = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9, cell10, cell11]
surface_list = [cube_x0, cube_x1, cube_y0, cube_y1, cube_z0, cube_z1, sp_left, sp_center, sp_right, pl_x0, pl_x1, pl_x2, pl_x3, cy, pl_rod0, pl_rod1]

#This function creates the cell geometry 
def create_cell_shape(cell, current_time):
    surface_geometry_shape_list = []
    
    for i in range(0,len(cell["surface_IDs"])):
        surface_ID = cell["surface_IDs"][i]
        if (len(surface_list[surface_ID].card["t"]) > 2):
            start_time = 0
            end_time = surface_list[surface_ID].card["t"][1]
            start_position = -surface_list[surface_ID].card["J"][0][0]
            end_position = -surface_list[surface_ID].card["J"][1][0]
            current_position = start_position + (((end_position-start_position)/(end_time-start_time))*current_time)
            
        else: 
            current_position = -surface_list[surface_ID].card["J"][0][0]
        if (surface_list[surface_ID].type == "plane-x"):
            point = ((current_position),0,0)
            if (not(cell["positive_flags"][i])):
                vector = (surface_list[surface_ID].card["G"], surface_list[surface_ID].card["H"], surface_list[surface_ID].card["I"])
            else:
                vector = (-surface_list[surface_ID].card["G"], surface_list[surface_ID].card["H"], surface_list[surface_ID].card["I"])
            surface_geometry_shape_list.append([Plane(Pnt(point), Vec(vector)),"intersect"])
          
           
        elif(surface_list[surface_ID].type == "plane-y"):
            point = (0, current_position,0)
            if (not(cell["positive_flags"][i])):
                vector = (surface_list[surface_ID].card["G"], surface_list[surface_ID].card["H"], surface_list[surface_ID].card["I"])
            else:
                vector = (surface_list[surface_ID].card["G"], -surface_list[surface_ID].card["H"], surface_list[surface_ID].card["I"])
            surface_geometry_shape_list.append([Plane(Pnt(point), Vec(vector)).col([1,0,0]), "intersect"])
        
        elif(surface_list[surface_ID].type == "plane-z"):
            point = (0,0,(current_position))
            if (not(cell["positive_flags"][i])):
                vector = (surface_list[surface_ID].card["G"], surface_list[surface_ID].card["H"], surface_list[surface_ID].card["I"])
            else:
                vector = (surface_list[surface_ID].card["G"], surface_list[surface_ID].card["H"], -surface_list[surface_ID].card["I"])
            surface_geometry_shape_list.append([Plane(Pnt(point), Vec(vector)).col([1,0,0]), "intersect"])
       
        elif(surface_list[surface_ID].type == "sphere"):
            x = surface_list[surface_ID].card["G"]/-2
            y = surface_list[surface_ID].card["H"]/-2
            z = surface_list[surface_ID].card["I"]/-2
            point = (x,y,z)
         
            radius =  float(math.sqrt(abs(surface_list[surface_ID].card["J"][0,0] - x**2 - y**2 - z**2)))

            if (not(cell["positive_flags"][i])):
                surface_geometry_shape_list.append([Sphere(Pnt(point), radius), "add"])
               
            else:
                surface_geometry_shape_list.append([Sphere(Pnt(point), radius), "subtract"])
   
            
        elif(surface_list[surface_ID].type == "cylinder-x"):
     
            y = surface_list[surface_ID].card["H"]/-2
            z = surface_list[surface_ID].card["I"]/-2

            radius = float(math.sqrt(abs(surface_list[surface_ID].card["J"][0,0] - z**2 - y**2)))
            
            if (not(cell["positive_flags"][i])):
                surface_geometry_shape_list.append([Cylinder(Pnt(-100,y,z), Pnt(100,y,z), radius),  "add"])
            else:
                surface_geometry_shape_list.append([Cylinder(Pnt(-100,y,z), Pnt(100,y,z), radius), "subtract"])
        elif(surface_list[surface_ID].type == "cylinder-y"):
     
            x = surface_list[surface_ID].card["G"]/-2
            z = surface_list[surface_ID].card["I"]/-2

            radius = float(math.sqrt(abs(surface_list[surface_ID].card["J"][0,0] - z**2 - y**2)))
            
            if (not(cell["positive_flags"][i])):
                surface_geometry_shape_list.append([Cylinder(Pnt(x,-100,z), Pnt(x,100,z), radius),  "add"])
            else:
                surface_geometry_shape_list.append([Cylinder(Pnt(x,-100,z), Pnt(x,100,z), radius), "subtract"])
        elif(surface_list[surface_ID].type == "cylinder-z"):
     
            x = surface_list[surface_ID].card["G"]/-2
            y = surface_list[surface_ID].card["H"]/-2

            radius = float(math.sqrt(abs(surface_list[surface_ID].card["J"][0,0] - z**2 - y**2)))
            
            if (not(cell["positive_flags"][i])):
                surface_geometry_shape_list.append([Cylinder(Pnt(x,y,-100), Pnt(x,y,100), radius),  "add"])
            else:
                surface_geometry_shape_list.append([Cylinder(Pnt(x,y,-100), Pnt(x,y,100), radius), "subtract"])
      
      
    for i in range(0, len(surface_geometry_shape_list)):
     
        if not(surface_geometry_shape_list[i][1] == "subtract"):
            if 'surface_geometry' not in locals():
                surface_geometry = surface_geometry_shape_list[i][0]
            elif surface_geometry_shape_list[i][1] == "intersect":

                surface_geometry = surface_geometry*surface_geometry_shape_list[i][0]
             
            elif surface_geometry_shape_list[i][1] == "add":
         
                surface_geometry = surface_geometry+surface_geometry_shape_list[i][0]
            #surface_geometry_shape_list.remove(surface_geometry_shape_list[i])

    for i in range(0, len(surface_geometry_shape_list)):
        if surface_geometry_shape_list[i][1] == "subtract":
     
                surface_geometry = surface_geometry-surface_geometry_shape_list[i][0]

    return surface_geometry

#This function is triggered when the slider is changed, 
#it creates new shapes for the cells based on the current time
def slider_changed(event):
    
    geo = CSGeometry() #create the geometry object
    for cell in cell_list:
    
        surface_geometry = create_cell_shape(cell, time_scale.get())

        if cell["material_ID"] == 1:
 
            geo.Add(surface_geometry.col([1,0,0]), transparent= True)

        else:
            geo.Add((surface_geometry.col([0,0,1])), transparent= True)
    geo.Draw()
    Redraw()

geo = CSGeometry() #create the geometry object


for cell in cell_list:
    surface_geometry = create_cell_shape(cell, 0)

    if cell["material_ID"] == 1:
        geo.Add(surface_geometry.col([1,0,0]), transparent= True)
    else:
        geo.Add((surface_geometry.col([0,0,1])), transparent= True)


geo.Draw()

Redraw()

#Set up slider
root = Tk()
root.title("Time Slider")
time_label = Label(root, text = "Time")
time_label.pack()
time_scale = Scale(root, from_=0, to=5, orient=HORIZONTAL, tickinterval=1, command=slider_changed)
time_scale.pack()

root.mainloop()

