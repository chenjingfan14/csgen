"""
Generating a REST inlet.

Author: Reece Otto 25/03/2022
"""
from csgen.stream_utils import inlet_stream_trace
from csgen.inlet_utils import inlet_blend
from nurbskit.path import Ellipse, Rectangle
from nurbskit.visualisation import path_plot_2D
import pyvista as pv

#------------------------------------------------------------------------------#
#                         Inlet defined by capture shape                       #
#------------------------------------------------------------------------------#
# generate capture shape
n_phi = 51
width = 1.0
height = width/2
centre_x = 0.0
centre_y = height/2 + 1/20
cap_shape = Rectangle(width, height, centre_x, centre_y)
cap_coords = cap_shape.list_eval(n_points=n_phi)

inlet_A_vals = {
    'job_name': 'trunc-bd',         # job name for puffin simulation
    'shape_coords': cap_coords,     # coordinates of capture shape
    'n_phi': n_phi,                 # number of phi points
    'n_z': 100,                     # number of z points
    'p_rat_shock': 2.0,             # pressure ratio for shock detection
    'dt': 1.0E-5,                   # integration time step [s]
    'max_step': 100000,             # maximum number of integration steps
    'plot_shape': True,             # option to plot capture shape
    'shape_label': 'Capture Shape', # figure label for capture shape
    'file_name_shape': 'cap_shape', # file name for capture shape plot
    'save_VTK': True,               # option to save inlet surface as VTK file
    'file_name_VTK': 'inlet_A'      # file name for VTK file
}

# run streamline tracer
inlet_A_surf = inlet_stream_trace(inlet_A_vals)

#------------------------------------------------------------------------------#
#                        Inlet defined by exit shape                           #
#------------------------------------------------------------------------------#
# generate exit shape
factor = 0.25
a_cap = 2*factor
b_cap = 1*factor
h_cap = 0.0
k_cap = centre_y
exit_shape = Ellipse(a_cap, b_cap, h_cap, k_cap)
exit_coords = exit_shape.list_eval(n_points=n_phi)

inlet_B_vals = {
    'job_name': 'trunc-bd',          # job name for puffin simulation
    'shape_coords': exit_coords,     # coordinates of capture shape
    'n_phi': n_phi,                  # number of phi points
    'n_z': 100,                      # number of z points
    'p_rat_shock': 2.0,              # pressure ratio for shock detection
    'dt': 1.0E-5,                    # integration time step [s]
    'max_step': 100000,              # maximum number of integration steps
    'plot_shape': True,              # option to plot exit shape
    'shape_label': 'Exit Shape',     # figure label for exit shape
    'file_name_shape': 'exit_shape', # file name for exit shape plot
    'save_VTK': True,                # option to save inlet surface as VTK file
    'file_name_VTK': 'inlet_B'       # file name for VTK file
}

# run streamline tracer
inlet_B_surf = inlet_stream_trace(inlet_B_vals)

#------------------------------------------------------------------------------#
#                               Blended inlet                                  #
#------------------------------------------------------------------------------#
inlet_C_coords = inlet_blend(inlet_A_surf, inlet_B_surf, 2.5)
inlet_grid_C = pv.StructuredGrid(inlet_C_coords[0], inlet_C_coords[1], 
        inlet_C_coords[2])
inlet_grid_C.save("inlet_C.vtk")