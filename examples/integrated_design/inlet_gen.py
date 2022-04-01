"""
Generating a shape-transition inlet.

Author: Reece Otto 25/03/2022
"""
from csgen.stream_utils import inlet_stream_trace
from csgen.inlet_utils import inlet_blend
from nurbskit.path import Ellipse, Rectangle
from nurbskit.visualisation import path_plot_2D
from nurbskit.geom_utils import rotate_x, translate
import pyvista as pv
import csv
import numpy as np

#------------------------------------------------------------------------------#
#                         Inlet defined by capture shape                       #
#------------------------------------------------------------------------------#
# generate capture shape
with open('exit_shape.csv', 'r') as csvfile:
    file = csv.reader(csvfile, delimiter=' ')
    coords = []
    for row in file:
        coords.append([float(row[0]), float(row[1])])
cap_coords = np.array(coords)
y_min = np.amin(cap_coords[:,1])
for i in range(len(cap_coords)):
    cap_coords[i][1] += abs(y_min) + 0.05
n_phi = len(cap_coords)

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
factor = 0.1
a_cap = 4*factor
b_cap = 1*factor
h_cap = 0.0
k_cap = 0.15
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

# translate and rotate for integration
z_plane = 5.0
z_min = np.amin(inlet_C_coords[:,:,2])
z_shift = z_plane - z_min
y_max = np.amin(inlet_C_coords[:,:,1])
y_attach = -0.6440822095774434
y_shift = -y_max + y_attach
inlet_C_coords = translate(inlet_C_coords, y_shift=y_shift, z_shift=z_shift)

"""
angle_attach = 7.34*pi/180
y_origin 
inlet_C_coords = rotate_x(inlet_C_coords, 7.34*pi/180, y_origin=0.0, 
    z_origin=z_plane)
"""

inlet_grid_C = pv.StructuredGrid(inlet_C_coords[:,:,0], inlet_C_coords[:,:,1], 
        inlet_C_coords[:,:,2])
inlet_grid_C.save("inlet_C.vtk")

