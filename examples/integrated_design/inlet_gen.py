"""
Generating a shape-transition inlet.

Author: Reece Otto 25/03/2022
"""
from csgen.stream_utils import inlet_stream_trace, shock_surface
from csgen.inlet_utils import inlet_blend
from nurbskit.path import Ellipse, Rectangle
from nurbskit.visualisation import path_plot_2D
from nurbskit.geom_utils import rotate_x, translate
import pyvista as pv
import csv
import numpy as np
from math import floor, pi
from scipy import interpolate, optimize
import os
import json

#------------------------------------------------------------------------------#
#                         Inlet defined by capture shape                       #
#------------------------------------------------------------------------------#
# generate capture shape
main_dir = os.getcwd()
os.chdir(main_dir + '/waverider')
with open('exit_shape.csv', 'r') as csvfile:
    file = csv.reader(csvfile, delimiter=' ')
    coords = []
    for row in file:
        coords.append([float(row[0]), float(row[1])])
os.chdir(main_dir)

cap_coords = np.array(coords)
y_min = np.amin(cap_coords[:,1])
for i in range(len(cap_coords)):
    cap_coords[i][1] += abs(y_min) + 0.05
n_phi = len(cap_coords)

working_dir = main_dir + '/inlet'
puffin_dir = main_dir + '/diffuser'
os.chdir(working_dir)

inlet_A_vals = {
    'puffin_dir': puffin_dir,       # job name for puffin simulation
    'job_name': 'diffuser',
    'shape_coords': cap_coords,     # coordinates of capture shape
    'n_phi': n_phi,                 # number of phi points
    'n_z': 100,                     # number of z points
    'n_r': 50,                      # number of r points for shock surface
    'p_rat_shock': 2.0,             # pressure ratio for shock detection
    'dt': 1.0E-5,                   # integration time step [s]
    'max_step': 100000,             # maximum number of integration steps
    'plot_shape': True,             # option to plot capture shape
    'shape_label': 'Capture Shape', # figure label for capture shape
    'file_name_shape': 'cap_shape', # file name for capture shape plot
    'save_VTK': False,              # option to save inlet surface as VTK file
    'file_name_VTK': 'inlet_A'      # file name for VTK file
}

# run streamline tracer
inlet_A_coords = inlet_stream_trace(inlet_A_vals)


"""
waverider = pv.read('waverider_surf.vtk').points
waverider = np.reshape(waverider, (51, 51, 3))
wr_trim = waverider[1:-1,:,:]
wr_trim_grid = pv.StructuredGrid(wr_trim[:,:,0], wr_trim[:,:,1], 
        wr_trim[:,:,2])
wr_trim_grid.save("wr_trim.vtk")

shock_interp = interpolate.Rbf(shock_surf[:,:,0], shock_surf[:,:,1], 
    shock_surf[:,:,2], epsilon=0.001)

wr_interp = interpolate.Rbf(wr_trim[:,:,0], wr_trim[:,:,1], 
    wr_trim[:,:,2], epsilon=0.001)

def residual(coords, surf_1, surf_2):
    x = coords[0]
    y = coords[1]
    return abs(surf_1(x,y) - surf_2(x,y))

x0 = [0.2, -0.7]
sol = optimize.minimize(residual, x0, method='SLSQP', 
    args=(shock_interp, wr_interp))
print(sol)
print(wr_interp(sol.x[0], sol.x[1]))
"""

#------------------------------------------------------------------------------#
#                        Inlet defined by exit shape                           #
#------------------------------------------------------------------------------#
# generate exit shape
factor = 0.1
a_cap = 2*factor
b_cap = 1*factor
h_cap = 0.0
k_cap = 0.15
exit_shape = Ellipse(a_cap, b_cap, h_cap, k_cap)
exit_coords = exit_shape.list_eval(n_points=n_phi)

inlet_B_vals = {
    'puffin_dir': puffin_dir,
    'job_name': 'diffuser',          # job name for puffin simulation
    'shape_coords': exit_coords,     # coordinates of capture shape
    'n_phi': n_phi,                  # number of phi points
    'n_z': 100,                      # number of z points
    'p_rat_shock': 2.0,              # pressure ratio for shock detection
    'dt': 1.0E-5,                    # integration time step [s]
    'max_step': 100000,              # maximum number of integration steps
    'plot_shape': True,              # option to plot exit shape
    'shape_label': 'Exit Shape',     # figure label for exit shape
    'file_name_shape': 'exit_shape', # file name for exit shape plot
    'save_VTK': False,               # option to save inlet surface as VTK file
    'file_name_VTK': 'inlet_B'       # file name for VTK file
}

# run streamline tracer
inlet_B_coords = inlet_stream_trace(inlet_B_vals)

#------------------------------------------------------------------------------#
#                               Blended inlet                                  #
#------------------------------------------------------------------------------#
inlet_C_coords = inlet_blend(inlet_A_coords, inlet_B_coords, 2.5)


f = open('attach.json')
attach_vals = json.load(f)
f.close()

angle_attach = attach_vals['attach_angle']
y_fb = attach_vals['y_attach']
z_fb = attach_vals['z_attach']

y_inlet = inlet_C_coords[floor(len(inlet_C_coords)/4),0,1]
z_inlet = inlet_C_coords[floor(len(inlet_C_coords)/4),0,2]

y_shift = -y_inlet + y_fb
z_shift = -z_inlet + z_fb

inlet_C_coords = translate(inlet_C_coords, y_shift=y_shift, z_shift=z_shift)

inlet_C_coords = rotate_x(inlet_C_coords, angle=angle_attach, x_origin=1.0,
    y_origin=y_fb, z_origin=z_fb)

inlet_A_coords = translate(inlet_A_coords, y_shift=y_shift, z_shift=z_shift)

inlet_A_coords = rotate_x(inlet_A_coords, angle=angle_attach, x_origin=1.0,
    y_origin=y_fb, z_origin=z_fb)

inlet_B_coords = translate(inlet_B_coords, y_shift=y_shift, z_shift=z_shift)

inlet_B_coords = rotate_x(inlet_B_coords, angle=angle_attach, x_origin=1.0,
    y_origin=y_fb, z_origin=z_fb)

inlet_grid_A = pv.StructuredGrid(inlet_A_coords[:,:,0], inlet_A_coords[:,:,1], 
        inlet_A_coords[:,:,2])
inlet_grid_A.save("inlet_A.vtk")

inlet_grid_B = pv.StructuredGrid(inlet_B_coords[:,:,0], inlet_B_coords[:,:,1], 
        inlet_B_coords[:,:,2])
inlet_grid_B.save("inlet_B.vtk")

inlet_grid_C = pv.StructuredGrid(inlet_C_coords[:,:,0], inlet_C_coords[:,:,1], 
        inlet_C_coords[:,:,2])
inlet_grid_C.save("inlet_C.vtk")

print('Constructing inlet shockwave surface.')
shock_surf = shock_surface(inlet_A_vals)
shock_surf = translate(shock_surf, y_shift=y_shift, z_shift=z_shift)
shock_surf = rotate_x(shock_surf, angle=angle_attach, x_origin=1.0,
    y_origin=y_fb, z_origin=z_fb)

shock_grid = pv.StructuredGrid(shock_surf[:,:,0], shock_surf[:,:,1], 
        shock_surf[:,:,2])
shock_grid.save("inlet_shock.vtk")




"""
ATTEMPTING INTERSECTION ALGORITHM HERE
ideas: 
 1 - create a curve in region of intersection and run point projection
     algorithm for each point on the curve
"""

from nurbskit.surface import BSplineSurface
from nurbskit.spline_fitting import global_surf_interp
os.chdir(main_dir + '/waverider')
wr_trim = np.reshape(pv.read('wr_trim.vtk').points, (49, 51, 3))
os.chdir(working_dir)


p = 3
q = 3
U, V, P = global_surf_interp(wr_trim, p, q)
wr_patch = BSplineSurface(p=p, q=q, U=U, V=V, P=P)

U, V, P = global_surf_interp(shock_surf, p, q)
shock_patch = BSplineSurface(p=p, q=q, U=U, V=V, P=P)

