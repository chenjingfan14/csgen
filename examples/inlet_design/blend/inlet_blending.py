"""
Using blending function to create an inlet with a rectangular capture shape and
elliptical exit shape.

Author: Reece Otto 23/02/2022
"""
from csgen.busemann import busemann_M1_p3p1
from csgen.atmosphere import atmos_interp
from csgen.shock_relations import beta_oblique, M2_oblique, p2_p1_oblique, \
                                  T2_T1_oblique
from csgen.stream_utils import busemann_stream_trace
from csgen.inlet_utils import inlet_blend
from nurbskit.path import Rectangle, Ellipse
from nurbskit.surface import BSplineSurface
from nurbskit.utils import auto_knot_vector
from nurbskit.visualisation import path_plot_2D
from nurbskit.spline_fitting import global_surf_interp
from nurbskit.cad_output import surf_to_vtk
from math import pi, ceil, floor
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

#------------------------------------------------------------------------------#
#                      Generating Busemann flow field                          #
#------------------------------------------------------------------------------#
# free-stream flow conditions
q0 = 50E3                                        # dynamic pressure [Pa]
M0 = 10                                          # Mach number
gamma = 1.4                                      # ratio of specific heats
p0 = 2 * q0 / (gamma * M0*M0)                    # static pressure [Pa]
T0 = atmos_interp(p0, 'Pressure', 'Temperature') # temperature [K]

# station 1 flow conditions (free-stream compressed by 6 deg wedge)
alpha = 6 * pi / 180                     # forebody wedge angle [rad]
beta = beta_oblique(alpha, M0, gamma)    # forebody shock angle [rad]
M1 = M2_oblique(beta, alpha, M0, gamma)  # Mach number
p1 = p2_p1_oblique(beta, M0, gamma) * p0 # static pressure [Pa]
T1 = T2_T1_oblique(beta, M0, gamma) * T0 # temperature [K]

print('Busemann flow field entrance conditions:')
print(f'M1 = {M1:.4}')
print(f'p1 = {p1:.4} Pa')
print(f'T1 = {T1:.4} K \n')

# Busemann flow field design parameters
p3 = 50E3             # desired exit pressure [Pa]
p3_p1 = p3/p1         # desired compression ratio
dtheta = 0.05*pi/180  # theta step size [rad]
n_streams = 51        # number of streamlines

# calculate Busemann flow field
field = busemann_M1_p3p1(M1, p3_p1, gamma, dtheta, print_freq=200)

# create plot of streamline
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
ax = plt.axes()
stream_coords = field.raw_stream
ax.plot(stream_coords[:,2], stream_coords[:,1], 'k-', label='Busemann Contour')
axis_coords = np.array([[np.amin(stream_coords[:,2]), 0],
                        [np.amax(stream_coords[:,2]), 0]])
ax.plot(axis_coords[:,0], axis_coords[:,1], 'k-.', label='Axis of Symmetry')
mw_coords = np.array([[stream_coords[:,2][-1], stream_coords[:,1][-1]],
                      [0, 0]])
ax.plot(mw_coords[:,0], mw_coords[:,1], 'r--', label='Entrance Mach Wave')
ts_coords = np.array([[0, 0],
                      [stream_coords[:,2][0], stream_coords[:,1][0]]])
ax.plot(ts_coords[:,0], ts_coords[:,1], 'r-', label='Terminating Shock Wave')
ax.set_xlabel('$z$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()

#------------------------------------------------------------------------------#
#         Generating inlet with rectangular capture shape (inlet A)            #
#------------------------------------------------------------------------------#
# generate shape of Busemann entrance with NURBS Ellipse
buse_ent = Ellipse(stream_coords[-1][1], stream_coords[-1][1])

# generate rectangular capture shape with a B-Spline path
width = 0.5
height = width / 2
centre_x = 0
centre_y = height/2 + stream_coords[-1][1] / 20
cap_shape = Rectangle(width, height, centre_x, centre_y)
cap_coords = cap_shape.list_eval(n_points=n_streams)

# plot capture shape on entrance of flow field
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
ax = path_plot_2D(cap_shape, show_control_points=False, show_control_net=False,
    show_knots=False, path_label='Capture Shape', path_style='r-')
ax = path_plot_2D(buse_ent, axes=ax, show_control_points=False, 
    show_control_net=False, show_knots=False, path_label='Busemann Entrance')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()
fig.savefig('capture_shape.svg', bbox_inches='tight')

# run streamline tracer and save the surface as a VTK file
inlet_A_coords = busemann_stream_trace(cap_coords, field, plane='capture')
inlet_grid_A = pv.StructuredGrid(inlet_A_coords[:,:,0], inlet_A_coords[:,:,1], 
        inlet_A_coords[:,:,2])
inlet_grid_A.save("inlet_A.vtk")

# save Busemann field as VTK file
buse_surf = field.surface(n_streams)
buse_grid = pv.StructuredGrid(buse_surf[0], buse_surf[1], buse_surf[2])
buse_grid.save("busemann.vtk")

# save Mach cone as VTK file
mc_surf = field.mach_cone_surface()
mc_grid = pv.StructuredGrid(mc_surf[0], mc_surf[1], mc_surf[2])
mc_grid.save("mach_cone.vtk")

# save terminating shock as VTK file
ts_surf = field.term_shock_surface()
ts_grid = pv.StructuredGrid(ts_surf[0], ts_surf[1], ts_surf[2])
ts_grid.save("terminating_shock.vtk")

#------------------------------------------------------------------------------#
#           Generating inlet with elliptical exit shape (inlet B)              #
#------------------------------------------------------------------------------#
# generate shape of Busemann exit with NURBS Ellipse
buse_exit = Ellipse(stream_coords[0][1], stream_coords[0][1])

# generate elliptical exit shape
a_exit = abs(inlet_A_coords[0][0][0])
b_exit = a_exit / 2
h_exit = 0
k_exit = b_exit + stream_coords[0][1] / 20
exit_shape = Ellipse(a_exit, b_exit, h_exit, k_exit)
exit_coords = exit_shape.list_eval(n_points=n_streams)

# plot exit shape on exit of flow field
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 20
    })
ax = path_plot_2D(buse_exit, show_control_points=False, 
    show_control_net=False,show_knots=False, path_label='Busemann Entrance')
ax = path_plot_2D(exit_shape, axes=ax, show_control_points=False, 
    show_control_net=False,show_knots=False, path_label='Inlet Exit Shape', 
    path_style='r-')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()
fig.savefig('exit_shape.svg', bbox_inches='tight')

# run streamline tracer and save the surface as a VTK file
inlet_B_coords = busemann_stream_trace(exit_coords, field, plane='exit')
inlet_grid_B = pv.StructuredGrid(inlet_B_coords[:,:,0], inlet_B_coords[:,:,1], 
        inlet_B_coords[:,:,2])
inlet_grid_B.save("inlet_B.vtk")

#------------------------------------------------------------------------------#
#                     Generating blended inlet (inlet C)                       #
#------------------------------------------------------------------------------#
# blend inlets A and B, then save the surface as a VTK file
inlet_C_coords = inlet_blend(inlet_A_coords, inlet_B_coords, 1/2.5)
inlet_grid_C = pv.StructuredGrid(inlet_C_coords[0], inlet_C_coords[1], 
        inlet_C_coords[2])
inlet_grid_C.save("inlet_C.vtk")