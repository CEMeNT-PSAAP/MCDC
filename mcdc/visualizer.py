from mcdc.print_ import print_warning

try:
    # launches visualization window
    # must be inside this loop so it doesn't launch when the visualizer is imported
    from netgen.meshing import *
    from netgen.csg import *
    from ngsolve import Draw, Redraw  # just for visualization
    import distinctipy  # creates unlimited visually distinct colors for visualization

except ImportError as e:
    msg = "\n >> MC/DC visualization error: \n >> dependencies for visualization not installed \n >> install optional dependencies needed for visualization with \n >>     <pip install mcdc['viz']> (for mac: 'mcdc[viz]')"
    print_warning(msg)

import tkinter as tk  # Tkinter is used to create the window for the time slider and color key
import math

# Get input_card and set global variables as "mcdc_"

import mcdc.global_ as mcdc_

input_card = mcdc_.input_deck


# get a point on the plane based on the current time in the system
# start and end times are zero by default
# called with a plane is visualized in create_cell_geometry()
def get_plane_current_position(surface, current_time, start_time, end_time):
    if len(surface["t"]) > 2:  # check if shape moves
        # establish reference points
        start_time = start_time  # default start time is zero
        end_time = end_time  # default end time is zero
        start_position = -surface["J"][0][0]
        end_position = -surface["J"][1][0]

        current_position = start_position + (
            ((end_position - start_position) / (end_time - start_time)) * current_time
        )

    else:
        current_position = -surface["J"][0][0]

    return current_position


# create the CSG geometry for a cell
# called by draw_geometry()
def create_cell_geometry(cell, current_time, surface_list, start_time, end_time):
    cell_shape_list = []  # list of shapes that make up the cell
    for i in range(0, len(cell["surface_IDs"])):
        surface_ID = cell["surface_IDs"][i]

        ## Check type of shape

        if surface_list[surface_ID]["type"] == "plane-x":
            # get reference point from the surface card
            point = (
                (
                    get_plane_current_position(
                        surface_list[surface_ID],
                        current_time,
                        start_time=start_time,
                        end_time=end_time,
                    )
                ),
                0,
                0,
            )

            # get normal vector from the surface card
            if cell["positive_flags"][i]:
                vector = (
                    -surface_list[surface_ID]["G"],
                    surface_list[surface_ID]["H"],
                    surface_list[surface_ID]["I"],
                )
            else:
                vector = (
                    surface_list[surface_ID]["G"],
                    surface_list[surface_ID]["H"],
                    surface_list[surface_ID]["I"],
                )

            # planes have to be intersected to achieve the wanted visualization in ngsolve
            cell_shape_list.append([Plane(Pnt(point), Vec(vector)), "intersect"])

        elif surface_list[surface_ID]["type"] == "plane-y":
            # get reference point from the surface card
            point = (
                0,
                (
                    get_plane_current_position(
                        surface_list[surface_ID],
                        current_time,
                        start_time=start_time,
                        end_time=end_time,
                    )
                ),
                0,
            )

            # get normal vector from the surface card
            if cell["positive_flags"][i]:
                vector = (
                    surface_list[surface_ID]["G"],
                    -surface_list[surface_ID]["H"],
                    surface_list[surface_ID]["I"],
                )
            else:
                vector = (
                    surface_list[surface_ID]["G"],
                    surface_list[surface_ID]["H"],
                    surface_list[surface_ID]["I"],
                )

            # planes have to be intersected to achieve the wanted visualization in ngsolve
            cell_shape_list.append(
                [Plane(Pnt(point), Vec(vector)).col([1, 0, 0]), "intersect"]
            )

        elif surface_list[surface_ID]["type"] == "plane-z":
            # get reference point from the surface card
            point = (
                0,
                0,
                (
                    get_plane_current_position(
                        surface_list[surface_ID],
                        current_time,
                        start_time=start_time,
                        end_time=end_time,
                    )
                ),
            )

            # get normal vector from the surface card
            if cell["positive_flags"][i]:
                vector = (
                    surface_list[surface_ID]["G"],
                    surface_list[surface_ID]["H"],
                    -surface_list[surface_ID]["I"],
                )
            else:
                vector = (
                    surface_list[surface_ID]["G"],
                    surface_list[surface_ID]["H"],
                    surface_list[surface_ID]["I"],
                )

            # planes have to be intersected to achieve the wanted visualization in ngsolve
            cell_shape_list.append(
                [Plane(Pnt(point), Vec(vector)).col([1, 0, 0]), "intersect"]
            )

        elif surface_list[surface_ID]["type"] == "sphere":
            # Get the center point from the surface card
            x = surface_list[surface_ID]["G"] / -2
            y = surface_list[surface_ID]["H"] / -2
            z = surface_list[surface_ID]["I"] / -2
            point = (x, y, z)

            # get radius from the surface card
            radius = float(
                math.sqrt(abs(surface_list[surface_ID]["J"][0, 0] - x**2 - y**2 - z**2))
            )

            # Add or subtract the sphere based on the CSG input
            if cell["positive_flags"][i]:
                cell_shape_list.append([Sphere(Pnt(point), radius), "subtract"])

            else:
                cell_shape_list.append([Sphere(Pnt(point), radius), "add"])

        elif surface_list[surface_ID]["type"] == "cylinder-x":
            # the y and z points are used to define the central axis of the cylinder
            y = surface_list[surface_ID]["H"] / -2
            z = surface_list[surface_ID]["I"] / -2

            # radius of the circle formed by a cross section parallel to the yz-plane
            radius = float(
                math.sqrt(abs(surface_list[surface_ID]["J"][0, 0] - z**2 - y**2))
            )

            # Add or subtract the sphere based on the CSG input
            # in the NGSolve CSG cylinders are infinite along their central axis,
            # by default we make the central axis -100<=x<=100
            # if a larger central axis is needed change below
            if cell["positive_flags"][i]:
                cell_shape_list.append(
                    [Cylinder(Pnt(-100, y, z), Pnt(100, y, z), radius), "subtract"]
                )
            else:
                cell_shape_list.append(
                    [Cylinder(Pnt(-100, y, z), Pnt(100, y, z), radius), "add"]
                )
        elif surface_list[surface_ID]["type"] == "cylinder-y":
            # the x and z points are used to define the central axis of the cylinder
            x = surface_list[surface_ID]["G"] / -2
            z = surface_list[surface_ID]["I"] / -2

            # radius of the circle formed by a cross section parallel to the xz-plane
            radius = float(
                math.sqrt(abs(surface_list[surface_ID]["J"][0, 0] - z**2 - y**2))
            )

            # Add or subtract the sphere based on the CSG input
            # in the NGSolve CSG cylinders are infinite along their central axis,
            # by default we make the central axis -100<=y<=100
            # if a larger central axis is needed change below
            if cell["positive_flags"][i]:
                cell_shape_list.append(
                    [Cylinder(Pnt(x, -100, z), Pnt(x, 100, z), radius), "subtract"]
                )
            else:
                cell_shape_list.append(
                    [Cylinder(Pnt(x, -100, z), Pnt(x, 100, z), radius), "add"]
                )
        elif surface_list[surface_ID]["type"] == "cylinder-z":
            # the x and y points are used to define the central axis of the cylinder
            x = surface_list[surface_ID]["G"] / -2
            y = surface_list[surface_ID]["H"] / -2

            # radius of the circle formed by a cross section parallel to the xy-plane
            radius = float(
                math.sqrt(abs(surface_list[surface_ID]["J"][0, 0] - z**2 - y**2))
            )

            # Add or subtract the sphere based on the CSG input
            # in the NGSolve CSG cylinders are infinite along their central axis,
            # by default we make the central axis -100<=z<=100
            # if a larger central axis is needed change below
            if cell["positive_flags"][i]:
                cell_shape_list.append(
                    [Cylinder(Pnt(x, y, -100), Pnt(x, y, 100), radius), "subtract"]
                )
            else:
                cell_shape_list.append(
                    [Cylinder(Pnt(x, y, -100), Pnt(x, y, 100), radius), "add"]
                )

    # combine the shapes for each cell
    # for subtraction to work properly, it must be done in the end
    for i in range(0, len(cell_shape_list)):
        if not (cell_shape_list[i][1] == "subtract"):
            if "cell_geometry" not in locals():
                cell_geometry = cell_shape_list[i][0]
            elif cell_shape_list[i][1] == "intersect":
                cell_geometry = cell_geometry * cell_shape_list[i][0]

            elif cell_shape_list[i][1] == "add":
                cell_geometry = cell_geometry + cell_shape_list[i][0]

    for i in range(0, len(cell_shape_list)):
        if cell_shape_list[i][1] == "subtract":
            cell_geometry = cell_geometry - cell_shape_list[i][0]

    return cell_geometry  # return the finished CSG geometry


# visualizes the model at a specified time (current_time, type float)
# called by visualize()
def draw_Geometry(current_time, start_time, end_time, material_colors):
    # create lists that contain all cells and surfaces
    surface_list = input_card.surfaces
    cell_list = input_card.cells

    geo = CSGeometry()  # create the ngsolve geometry object

    # colors that should not be generated  by distinctipy(starts with visually unappealing colors, manually set colors added later)
    # These colors are rgb values, more can be added by extending the list
    input_colors = [
        (1, 1, 1),  # white
        (0, 0, 0),  # black
    ]
    # list of materials that need colors to be generated (ie not water or the source)
    material_colors_to_generate = []

    # if the color of water and source are not set make them blue and green respectively
    # add manually specified colors to input colors.
    for cell in cell_list:
        cell_material_name = cell["material_name"]
        if cell_material_name not in list(material_colors.keys()):
            if cell_material_name == "water":
                material_colors["water"] = [0, 0, 1]
                input_colors.append((0, 0, 1))
            elif cell_material_name == "source":
                material_colors["source"] = [0, 1, 0]
                input_colors.append((0, 1, 0))
            else:
                material_colors[cell_material_name] = None
                material_colors_to_generate.append(cell_material_name)

    # create n number of distinct colors where n
    # is the number of materials in material_colors_to_generate
    distinct_colors = distinctipy.get_colors(
        len(material_colors_to_generate), input_colors
    )
    for i in range(0, len(material_colors_to_generate)):
        material_colors[material_colors_to_generate[i]] = [
            distinct_colors[i][0],
            distinct_colors[i][1],
            distinct_colors[i][2],
        ]

    # cycle through the cells in the model
    for cell_index in range(0, len(cell_list)):
        cell = cell_list[cell_index]

        # create the geometry for the cell
        cell_geometry = create_cell_geometry(
            cell,
            current_time=current_time,
            surface_list=surface_list,
            start_time=start_time,
            end_time=end_time,
        )
        # add the cell geometry to the visualization
        geo.Add(
            cell_geometry.col(material_colors[cell["material_name"]]), transparent=True
        )

    # draw the visualization
    geo.Draw()
    Redraw()

    return material_colors


# displays the color key to the user
# called by visualize()
def create_color_key(root, color_key_dict):
    tk.Label(root, text="color key").grid(row=2, column=0)

    # for each material in the color_key_dict display
    # the material name and corresponding color to the user
    for material_index in range(0, len(color_key_dict)):
        # create label for the material name
        tk.Label(root, text=str(list(color_key_dict)[material_index]) + ":").grid(
            row=3 + material_index, sticky=tk.W
        )

        # canvas where color will be displayed
        canvas = tk.Canvas(root, width=200, height=len(color_key_dict) * 50)

        # switch from rgb to hex for tkinter
        rgb = list(color_key_dict.values())[material_index]
        rgb = [int(255 * rgb[0]), int(255 * rgb[1]), int(255 * rgb[2])]
        colorval = "#{0:02x}{1:02x}{2:02x}".format(rgb[0], rgb[1], rgb[2])

        # add rectangle to canvas with material color
        canvas.create_rectangle(10, 10, 70, 60, fill=colorval)

        # add canvas to the window
        canvas.grid(row=3 + material_index, column=1, sticky=tk.E)


# triggered when time slider or time spinbox changed
# it redraws the model at the new time
def time_slider_changed(current_time, start_time, end_time, material_colors):
    draw_Geometry(
        current_time=float(current_time),
        start_time=start_time,
        end_time=end_time,
        material_colors=material_colors,
    )


# creates the time slider
# called by visualize()
def create_time_slider(root, start_time, end_time, tick_interval, material_colors):
    root.title("Time Slider")

    time_label = tk.Label(root, text="Time")
    time_label.grid(row=0, column=0, columnspan=2)

    time_var = tk.StringVar(root, "0")

    time_scale = tk.Scale(
        root,
        from_=start_time,
        to=end_time,
        orient=tk.HORIZONTAL,
        resolution=tick_interval,
        variable=time_var,
        command=lambda event: time_slider_changed(
            event, start_time, end_time, material_colors=material_colors
        ),
        length=400,
    )
    time_scale.grid(row=1, column=1, columnspan=4)

    time_spinbox = tk.Spinbox(
        root,
        from_=start_time,
        to=end_time,
        textvariable=time_var,
        increment=tick_interval,
        command=lambda: time_slider_changed(
            time_var.get(), start_time, end_time, material_colors=material_colors
        ),
    )
    time_spinbox.grid(row=0, column=3)


# runs the visualization for a model
# start and end times are default zero
# called in input file
def visualize(start_time=0, end_time=0, tick_interval=1, material_colors={}):
    # The dependence for the visualizer are quite large and include
    # netgen (75MB), intel mkl (200MB), among others
    # These are configurable as optional dependencies
    try:
        # launches visualization window
        # must be inside this loop so it doesn't launch when the visualizer is imported
        import netgen.gui
    except ImportError as e:
        msg = "\n >> MC/DC visualization error: \n >> dependencies for visualization not installed \n >> install optional dependencies needed for visualization with \n >>     <pip install mcdc['viz']> (for mac: 'mcdc[viz]')"
        print_warning(msg)

    color_key_dic = draw_Geometry(
        current_time=0,
        start_time=start_time,
        end_time=end_time,
        material_colors=material_colors,
    )

    # Set up tkinter window
    root = tk.Tk()
    if start_time != end_time:
        create_time_slider(root, start_time, end_time, tick_interval, color_key_dic)
    create_color_key(root, color_key_dic)
    root.mainloop()  # mainloop for tkinter window
