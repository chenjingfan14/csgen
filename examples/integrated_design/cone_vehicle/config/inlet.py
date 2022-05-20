"""
Generating a shape-transition inlet.

Author: Reece Otto 25/03/2022
"""
from csgen.stream_utils import inlet_stream_trace
from csgen.inlet_utils import inlet_blend
from csgen.grid import StructuredGrid
from nurbskit.path import Ellipse
from nurbskit.geom_utils import rotate_x, translate
from nurbskit.spline_fitting import global_surf_interp
from nurbskit.surface import BSplineSurface
from nurbskit.cad_output import nurbs_surf_to_iges
import os
import csv
import numpy as np
import json

#------------------------------------------------------------------------------#
#                         Inlet Defined by Capture Shape                       #
#------------------------------------------------------------------------------#
# import capture shape
config_dir = os.getcwd()
main_dir = os.path.dirname(config_dir)
diffuser_dir = main_dir + '/diffuser'
inlet_dir = main_dir + '/inlet'
os.chdir(inlet_dir)

with open('cap_shape_adj.csv', 'r') as csvfile:
    file = csv.reader(csvfile, delimiter=' ')
    next(file)
    coords = []
    for row in file:
        coords.append([float(row[0]), float(row[1])])

cap_coords = np.array(coords)
y_min = np.amin(cap_coords[:,1])
cap_shifted = cap_coords.copy()
for i in range(len(cap_coords)):
    cap_shifted[i][1] += abs(y_min) + 0.05
n_phi = len(cap_coords)

# run streamline tracer
inlet_A_vals = {
    'puffin_dir': diffuser_dir,     # directory of puffin simulation
    'job_name': 'diffuser',         # job name for puffin simulation
    'shape_coords': cap_shifted,    # coordinates of capture shape
    'n_phi': n_phi,                 # number of phi points
    'n_z': 100,                     # number of z points
    'p_rat_shock': 2.0,             # pressure ratio for shock detection
    'dt': 1.0E-5,                   # integration time step [s]
    'max_step': 100000,             # maximum number of integration steps
    'plot_shape': True,             # option to plot capture shape
    'shape_label': 'Capture Shape', # figure label for capture shape
    'file_name_shape': 'cap_shape', # file name for capture shape plot
    'save_VTK': False,              # option to save inlet surface as VTK file
    'file_name_VTK': 'inlet_A'      # file name for VTK file
}
inlet_A = inlet_stream_trace(inlet_A_vals)
StructuredGrid(inlet_A).export_to_vtk_xml(file_name='inlet_A')

#------------------------------------------------------------------------------#
#                        Inlet Defined by Exit Shape                           #
#------------------------------------------------------------------------------#
# generate exit shape
factor = 0.3
a_cap = 2*factor
b_cap = 1*factor
h_cap = 0.0
k_cap = 0.5
exit_shape = Ellipse(a_cap, b_cap, h_cap, k_cap)
exit_coords = exit_shape.list_eval(n_points=n_phi)

# run streamline tracer
inlet_B_vals = inlet_A_vals.copy()
inlet_B_vals['shape_coords'] = exit_coords.copy()
inlet_B_vals['shape_label'] = 'Exit Shape'
inlet_B_vals['file_name_VTK'] = 'inlet_B'
inlet_B = inlet_stream_trace(inlet_B_vals)
StructuredGrid(inlet_B).export_to_vtk_xml(file_name='inlet_B')

#------------------------------------------------------------------------------#
#                               Blended Inlet                                  #
#------------------------------------------------------------------------------#
inlet_C = inlet_blend(inlet_A, inlet_B, 2.5)

#------------------------------------------------------------------------------#
#                     Transform Inlets to Correct Orientation                  #                               #
#------------------------------------------------------------------------------#
# import forebody attachment info
os.chdir(diffuser_dir)
f = open('inlet_inflow.json')
attach_vals = json.load(f)
f.close()
os.chdir(inlet_dir)
angle_attach = attach_vals['theta']
y_fb = attach_vals['y']
z_fb = attach_vals['z']

# calculate required translation
y_inlet = inlet_C[len(inlet_C)//4,0,1]
z_inlet = inlet_C[len(inlet_C)//4,0,2]

y_shift = -y_inlet + y_fb
z_shift = -z_inlet + z_fb

inlet_A = translate(inlet_A, y_shift=y_shift, z_shift=z_shift)
inlet_A = rotate_x(inlet_A, angle=angle_attach, x_origin=1.0,
    y_origin=y_fb, z_origin=z_fb)

inlet_B = translate(inlet_B, y_shift=y_shift, z_shift=z_shift)
inlet_B = rotate_x(inlet_B, angle=angle_attach, x_origin=1.0,
    y_origin=y_fb, z_origin=z_fb)

inlet_C = translate(inlet_C, y_shift=y_shift, z_shift=z_shift)
inlet_C = rotate_x(inlet_C, angle=angle_attach, x_origin=1.0,
    y_origin=y_fb, z_origin=z_fb)


#------------------------------------------------------------------------------#
#                  Ensure Integration Interface is Watertight                  #                               #
#------------------------------------------------------------------------------#
with open('intersect_contour.csv', 'r') as csvfile:
    file = csv.reader(csvfile, delimiter=' ')
    next(file)
    intersect_cont = []
    for row in file:
        intersect_cont.append([float(row[0]), float(row[1]), float(row[2])])

inlet_A[:,0][len(inlet_A[:,0])//8:3*len(inlet_A[:,0])//8+1] = intersect_cont
inlet_C[:,0][len(inlet_A[:,0])//8:3*len(inlet_A[:,0])//8+1] = intersect_cont

StructuredGrid(inlet_A).export_to_vtk_xml(file_name='inlet_A')
StructuredGrid(inlet_B).export_to_vtk_xml(file_name='inlet_B')
StructuredGrid(inlet_C).export_to_vtk_xml(file_name='inlet_C')

p_inlet = 3
U_top, V_top, P_top = global_surf_interp(inlet_C[:len(inlet_C)//2+1], p_inlet, 
                                         p_inlet)
inlet_top = BSplineSurface(p=p_inlet, q=p_inlet, U=U_top, V=V_top, P=P_top)
inlet_top = inlet_top.cast_to_nurbs_surface()
nurbs_surf_to_iges(inlet_top, file_name='inlet_top')

U_bot, V_bot, P_bot = global_surf_interp(inlet_C[len(inlet_C)//2:], p_inlet, 
                                         p_inlet)
inlet_bot = BSplineSurface(p=p_inlet, q=p_inlet, U=U_bot, V=V_bot, P=P_bot)
inlet_bot = inlet_bot.cast_to_nurbs_surface()
nurbs_surf_to_iges(inlet_bot, file_name='inlet_bot')


"""
TO-DO
 - allow for multiple engine modules
 - allow for capture shapes where n_i and n_j are different
 - streamline trace backwards from throat for exit shape
"""