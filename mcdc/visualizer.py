#Visualization imports
import netgen.gui #launches visualiztation window

from ngsolve import Draw, Redraw # just for visualization

from netgen.meshing import *
from netgen.csg import *

import math

from tkinter import * #Tkinter is used to create the window for the time slider and color key
import distinctipy #creates unlimited visually distinct colors for visualization


# Get input_card and set global variables as "mcdc_"
import mcdc.global_ as mcdc_

input_card = mcdc_.input_card

#get a point on the plane based on the current time in the system
#start and end times are zero by default
#called with a plane is visualized in create_cell_geometry()
def get_plane_current_position(surface, current_time, start_time, end_time):
    
    if (len(surface["t"]) > 2):#check if shape moves
        #establish reference points
        start_time = start_time #default start time is zero
        end_time = end_time
        start_position = -surface["J"][0][0]
        end_position = -surface["J"][1][0]

        current_position = start_position + (((end_position-start_position)/(end_time-start_time))*current_time)

    else: 
        current_position = -surface["J"][0][0]

    return current_position

    
#create the CSG geometry for a cell
#called by draw_geometry()
def create_cell_geometry(cell, current_time, surface_list, start_time, end_time):
    cell_shape_list = [] #list of shapes that make up the cell
    for i in range(0,len(cell["surface_IDs"])):

        surface_ID = cell["surface_IDs"][i]

        # Check type of shape    
        if (surface_list[surface_ID]["type"] == "plane-x"):

            #get reference point from the surface card
            point = ((get_plane_current_position(surface_list[surface_ID], current_time, start_time=start_time, end_time=end_time)),0,0)

            #get normal vector from the surface card
            if (cell["positive_flags"][i]):
                vector = (-surface_list[surface_ID]["G"], 
                          surface_list[surface_ID]["H"], 
                          surface_list[surface_ID]["I"])
            else:
                vector = (surface_list[surface_ID]["G"], 
                          surface_list[surface_ID]["H"], 
                          surface_list[surface_ID]["I"])
                
            # planes have to be intersected to achive the wanted visualization in ngsolve
            cell_shape_list.append([Plane(Pnt(point), Vec(vector)),"intersect"]) 
           
        elif(surface_list[surface_ID]["type"] == "plane-y"):

            #get reference point from the surface card
            point = (0, (get_plane_current_position(surface_list[surface_ID], current_time, start_time=start_time, end_time=end_time)),0)

            #get normal vector from the surface card
            if (cell["positive_flags"][i]):
                vector = (surface_list[surface_ID]["G"], 
                          -surface_list[surface_ID]["H"], 
                          surface_list[surface_ID]["I"])
            else:
                vector = (surface_list[surface_ID]["G"], 
                          surface_list[surface_ID]["H"], 
                          surface_list[surface_ID]["I"])

            # planes have to be intersected to achive the wanted visualization in ngsolve
            cell_shape_list.append([Plane(Pnt(point), Vec(vector)).col([1,0,0]), "intersect"]) 
        
        elif(surface_list[surface_ID]["type"] == "plane-z"):

            #get reference point from the surface card
            point = (0,0,(get_plane_current_position(surface_list[surface_ID], current_time, start_time=start_time, end_time=end_time)))

            #get normal vector from the surface card
            if (cell["positive_flags"][i]):
                vector = (surface_list[surface_ID]["G"], 
                          surface_list[surface_ID]["H"], 
                          -surface_list[surface_ID]["I"])
            else:
                vector = (surface_list[surface_ID]["G"], 
                          surface_list[surface_ID]["H"], 
                          surface_list[surface_ID]["I"])

            # planes have to be intersected to achive the wanted visualization in ngsolve
            cell_shape_list.append([Plane(Pnt(point), Vec(vector)).col([1,0,0]), "intersect"]) 
       
        elif(surface_list[surface_ID]["type"] == "sphere"):

            #Get the center point from the surface card
            x = surface_list[surface_ID]["G"]/-2
            y = surface_list[surface_ID]["H"]/-2
            z = surface_list[surface_ID]["I"]/-2
            point = (x,y,z)
         
            #get radius from the surface card
            radius =  float(math.sqrt(abs(surface_list[surface_ID]["J"][0,0] - x**2 - y**2 - z**2)))

            #Add or subtract the sphere based on the CSG input
            if ((cell["positive_flags"][i])):
                cell_shape_list.append([Sphere(Pnt(point), radius), "subtract"])
               
            else:
                cell_shape_list.append([Sphere(Pnt(point), radius), "add"])
            
        elif(surface_list[surface_ID]["type"] == "cylinder-x"):
     
            #the y and z points are used to define the central axis of the cylinder
            y = surface_list[surface_ID]["H"]/-2
            z = surface_list[surface_ID]["I"]/-2

            #radius of the circle formed by a cross section parallel to the yz-plane
            radius = float(math.sqrt(abs(surface_list[surface_ID]["J"][0,0] - z**2 - y**2)))
            
            #Add or subtract the sphere based on the CSG input
            #in the NGSolve CSG cylinders are infinite along their central axis, 
            # by default we make the central axis -100<=x<=100
            #if a larger central axis is needed change below
            if ((cell["positive_flags"][i])):
                cell_shape_list.append([Cylinder(Pnt(-100,y,z), Pnt(100,y,z), radius),  "subtract"])
            else:
                cell_shape_list.append([Cylinder(Pnt(-100,y,z), Pnt(100,y,z), radius), "add"])
        elif(surface_list[surface_ID]["type"] == "cylinder-y"):
     
            #the x and z points are used to define the central axis of the cylinder
            x = surface_list[surface_ID]["G"]/-2
            z = surface_list[surface_ID]["I"]/-2

            #radius of the circle formed by a cross section parallel to the xz-plane
            radius = float(math.sqrt(abs(surface_list[surface_ID]["J"][0,0] - z**2 - y**2)))
            
            #Add or subtract the sphere based on the CSG input
            #in the NGSolve CSG cylinders are infinite along their central axis, 
            # by default we make the central axis -100<=y<=100
            #if a larger central axis is needed change below
            if ((cell["positive_flags"][i])):
                cell_shape_list.append([Cylinder(Pnt(x,-100,z), Pnt(x,100,z), radius),  "subtract"])
            else:
                cell_shape_list.append([Cylinder(Pnt(x,-100,z), Pnt(x,100,z), radius), "add"])
        elif(surface_list[surface_ID]["type"] == "cylinder-z"):
     
            #the x and y points are used to define the central axis of the cylinder
            x = surface_list[surface_ID]["G"]/-2
            y = surface_list[surface_ID]["H"]/-2

            #radius of the circle formed by a cross section parallel to the xy-plane
            radius = float(math.sqrt(abs(surface_list[surface_ID]["J"][0,0] - z**2 - y**2)))
            
            #Add or subtract the sphere based on the CSG input
            #in the NGSolve CSG cylinders are infinite along their central axis, 
            # by default we make the central axis -100<=z<=100
            #if a larger central axis is needed change below
            if ((cell["positive_flags"][i])):
                cell_shape_list.append([Cylinder(Pnt(x,y,-100), Pnt(x,y,100), radius),  "subtract"])
            else:
                cell_shape_list.append([Cylinder(Pnt(x,y,-100), Pnt(x,y,100), radius), "add"])
      

    #combine the shapes for each cell
    #for subtraction to work properly, it must be done in the end
    for i in range(0, len(cell_shape_list)):
     
        if not(cell_shape_list[i][1] == "subtract"):
            if 'cell_geometry' not in locals():
                cell_geometry = cell_shape_list[i][0]
            elif cell_shape_list[i][1] == "intersect":

                cell_geometry = cell_geometry*cell_shape_list[i][0]
             
            elif cell_shape_list[i][1] == "add":
         
                cell_geometry = cell_geometry+cell_shape_list[i][0]

    for i in range(0, len(cell_shape_list)):
        if cell_shape_list[i][1] == "subtract":
     
                cell_geometry = cell_geometry-cell_shape_list[i][0]

    return cell_geometry #return the finished CSG geometry

       
# visualizes the model at a specified time (current_time, type float)
#called by visualize()
def draw_Geometry(current_time, start_time, end_time):

    #create lists that contain all cells and surfaces
    cell_list = input_card.cells
    surface_list = input_card.surfaces

    #make water blue and the source green
    water_rgb = [0,0,1]
    source_rgb = [0,1,0]

    geo = CSGeometry() #create the ngsolve geometry object
 
    #list of materials that need colors to be generated (ie not water or the source)
    material_colors_to_generate = [] 

    # find the materials that need colors generated and add them to material_colors_to_generate
    for cell in cell_list:
        cell_material_name = cell["material_name"]
        if (
            (cell_material_name not in material_colors_to_generate) and 
            (cell_material_name != "water") and
            (cell_material_name != "source")
        ):
            material_colors_to_generate.append(cell_material_name)
 

    #colors that should not be generated (ie, taken by preset materials or which are visually unappealing)
    #These colors are rgb values, more can be added by extending the list
    input_colors = [
                    (water_rgb[0], water_rgb[1], water_rgb[2]), #water - blue
                    (source_rgb[0], source_rgb[1], source_rgb[2]), #source - green
                    (1,1,1), #white
                    (0,0,0) #black
                    ]
    #create n number of distinct colors where n is the number of materials in material_colors_to_generate
    distinct_colors = distinctipy.get_colors(len(material_colors_to_generate), input_colors)
 
    #This list will later be passed to create_color_key 
    #contains lists of format [rgb value, material name]
    color_key_list = [] 
    materials_added_to_color_key = []

    #cycle through the cells in the model
    for cell_index in range(0, len(cell_list)):
        cell = cell_list[cell_index]

        #create the geometry for the cell
        cell_geometry = create_cell_geometry(cell, current_time= current_time, surface_list=surface_list, start_time=start_time, end_time=end_time)

        #assign the material an rgb value
        cell_material_name = cell["material_name"]
        if cell_material_name == "water":
            rgb = water_rgb
        elif cell_material_name == "source":
            rgb = source_rgb
        else:
            rgb = distinct_colors[material_colors_to_generate.index(cell_material_name)]
            rgb = [int(rgb[0]), int(rgb[1]),int(rgb[2])] #ngsolve takes rgb as a list
           
        #if material is missing from the color key, add it
        if cell_material_name not in materials_added_to_color_key:
            materials_added_to_color_key.append(cell_material_name)
            color_key_list.append([rgb, cell_material_name])

        #add the cell geometry to the visualization
        geo.Add(cell_geometry.col(rgb), transparent=True)

    #draw the visualization
    geo.Draw()
    Redraw()
    
    return color_key_list


#displays the color key to the user
#called by visualize()
def create_color_key(root, color_key_list):
    Label(root, text = "color key").grid(row = 2, column = 0)

    #for each material in the color_key_list display 
    # the material name and corresponding color to the user
    for material_index in range(0, len(color_key_list)):
        material = color_key_list[material_index]

        #create label for the material name
        Label(root, text= str(material[1])+ ":").grid(row = 3 +material_index, sticky=W)
        
        #canvas where color will be displayed
        canvas = Canvas(root, width= 200, height = len(color_key_list)*50) 

        #switch from rgb to hex for tkinter
        rgb = material[0]
        rgb = [255*rgb[0], 255*rgb[1], 255*rgb[2]]
        colorval = '#{0:02x}{1:02x}{2:02x}'.format(rgb[0],rgb[1],rgb[2]) 
     
        # add rectangle to canvas with material color
        canvas.create_rectangle(10, 10,70, 60,fill =colorval) 

        #add canvas to the window
        canvas.grid(row = 3 + material_index, column = 1, sticky=E) 
 
#triggered when time slider changed
#it redraws the model at the new time
def time_slider_changed(event, start_time, end_time):
    draw_Geometry(current_time=float(event), start_time= start_time, end_time=end_time)

#creates the time slider
#called by visualize()
def create_time_slider(root, start_time, end_time):
    root.title("Time Slider")
    
    time_label = Label(root, text = "Time")
    time_label.grid(row = 0, column = 0, columnspan=2)

    time_scale = Scale(root, from_=start_time, to=end_time, orient=HORIZONTAL, tickinterval=1, command=lambda event: time_slider_changed(event,start_time, end_time ))
    time_scale.grid(row = 1, column = 1)

#runs the visualization for a model
#start and end times are default zero
#called in input file
def visualize(start_time = 0, end_time = 0):
    
    color_key_list = draw_Geometry(current_time=0, start_time=start_time, end_time=end_time)
    

    #Set up tkinter window
    root = Tk()
    if (start_time != end_time):
        create_time_slider(root, start_time, end_time)
    create_color_key(root, color_key_list )
    root.mainloop() #mainloop for tkinter window



