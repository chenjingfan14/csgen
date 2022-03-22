"""
Streamline tracing a waverider exit shape through a Busemann flow field.

Author: Reece Otto 07/03/2022
"""
from csgen.busemann import busemann_M1_p3p1
from csgen.stream_utils import busemann_stream_trace
from nurbskit.path import Ellipse
from nurbskit.surface import BSplineSurface
from nurbskit.geom_utils import translate, rotate_x
from nurbskit.spline_fitting import global_surf_interp
from nurbskit.cad_output import nurbs_surf_to_iges
import pyvista as pv
from math import pi, sqrt, tan, floor
import matplotlib.pyplot as plt
import numpy as np
import csv

# Busemann flow field design parameters
M1 = 8.233   # entrance Mach number
p1 = 2.269E3 # entrance static pressure [Pa]
T1 = 301.7   # entrance temperature [K]
gamma = 1.4  # ratio of specific heats

p3 = 50E3              # desired exit pressure [Pa]
p3_p1 = p3/p1          # desired compression ratio
dtheta = 0.09 * pi/180 # theta step size [rad]
n_streams = 51        # number of streamlines
z_plane = 5            # z-coordinate of waverider exit plane
theta_attach = 7.34*pi/180 # inlet attachment angle [rad]

# import capture shape
with open('exit_shape.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    cap_coords = np.nan * np.ones((len(list(csv_reader)), 2))

with open('exit_shape.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    i = 0
    for row in csv_reader:
        cap_coords[i][0] = row[0]
        cap_coords[i][1] = row[1]
        i += 1

min_y = np.amin(cap_coords[:,1])
y_shift = abs(min_y) + 0.01
for i in range(len(cap_coords)):
    cap_coords[i][1] += y_shift

# generate Busemann field that fits capture shape in top half
field = busemann_M1_p3p1(M1, p3_p1, gamma, dtheta, beta2_guess=0.2312, 
    M2_guess=5.45, print_freq=1000)
r_ind = np.argmin(cap_coords[:,0])
x_cap_max_r = np.amin(cap_coords[:,0])
y_cap_max_r = cap_coords[:,1][r_ind]
cap_max_r = sqrt(x_cap_max_r**2 + y_cap_max_r**2)
r0_scaled = cap_max_r / field.ys[-1]
field = busemann_M1_p3p1(M1, p3_p1, gamma, dtheta, r0=r0_scaled,
    beta2_guess=0.2312, M2_guess=5.45, print_freq=1000)

# create plot of streamline
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 20
    })
ax = plt.axes()
ax.plot(field.zs, field.ys, 'k-', label='Busemann Contour')
ax.plot([field.zs[0], field.zs[-1]], [0, 0], 'k-.', label='Axis of Symmetry')
ax.plot([field.zs[0], 0], [field.ys[0], 0], 'r--', label='Entrance Mach Wave')
ax.plot([0, field.zs[-1]], [0, field.ys[-1]], 'r-', 
    label='Terminating Shock Wave')
ax.set_xlabel('$z$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()
fig.savefig('busemann.svg', bbox_inches='tight')

# plot capture shape on entrance of flow field
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 20
    })
ax = plt.axes()
buse_ent = Ellipse(field.ys[-1], field.ys[-1])
buse_ent_coords = buse_ent.list_eval(n_points=n_streams)
ax.plot(buse_ent_coords[:,0], buse_ent_coords[:,1], 'k-', 
        label='Busemann Entrance')
ax.plot(cap_coords[:,0], cap_coords[:,1], 'r-', label='Inlet Capture Shape')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()
fig.savefig('capture_shape.svg', bbox_inches='tight')

# run streamline tracer and re-position inlet to exit plane of waverider
inlet_surf = busemann_stream_trace(cap_coords, field)
# translate
#z_attach = np.amin(inlet_surf[:,:,2])
z_attach = inlet_surf[:,:,2][floor(len(inlet_surf[:,:,2])/4)][-1]
z_shift = z_plane + abs(z_attach)
inlet_surf = translate(inlet_surf, y_shift=-y_shift, z_shift=z_shift)
# rotate
y_origin = inlet_surf[:,:,1][floor(len(inlet_surf[:,:,2])/4)][-1]
z_origin = inlet_surf[:,:,2][floor(len(inlet_surf[:,:,2])/4)][-1]
inlet_surf = rotate_x(inlet_surf, theta_attach, x_origin=1, y_origin=y_origin, 
    z_origin=z_plane)

"""
# fit b-spline surface to inlet and export as IGES
print('\nFitting B-Spline surface to inlet.')
p = 2
q = 3
U, V, P = global_surf_interp(inlet_surf, p, q)
inlet_spline = BSplineSurface(p=p, q=q, U=U, V=V, P=P)
inlet_spline = inlet_spline.cast_to_nurbs_surface()
nurbs_surf_to_iges(inlet_spline, file_name='inlet')

# save full inlet grid as vtk
inlet_grid = pv.StructuredGrid(inlet_surf[:,:,0], inlet_surf[:,:,1], 
    inlet_surf[:,:,2])
inlet_grid.save("waverider_inlet.vtk")

# save coarse inlet grid as VTK
inlet_coarse = inlet_spline.list_eval(N_u=50, N_v=50)
inlet_coarse = pv.StructuredGrid(inlet_coarse[:,:,0], inlet_coarse[:,:,1], 
    inlet_coarse[:,:,2])
inlet_coarse.save("inlet_coarse.vtk")
"""

# save Busemann field as VTK file
buse_surf = field.surface(n_streams)
buse_surf = translate(buse_surf, y_shift=-y_shift, z_shift=z_shift)
buse_surf = rotate_x(buse_surf, theta_attach, x_origin=1, y_origin=y_origin, 
    z_origin=z_plane)
buse_grid = pv.StructuredGrid(buse_surf[:,:,0], buse_surf[:,:,1], 
    buse_surf[:,:,2])
buse_grid.save("busemann.vtk")

# save Mach cone as VTK file
mc_surf = field.mach_cone_surface()
mc_surf = translate(mc_surf, y_shift=-y_shift, z_shift=z_shift)
mc_surf = rotate_x(mc_surf, theta_attach, x_origin=1, y_origin=y_origin, 
    z_origin=z_plane)
mc_grid = pv.StructuredGrid(mc_surf[:,:,0], mc_surf[:,:,1], mc_surf[:,:,2])
mc_grid.save("mach_cone.vtk")

# save terminating shock as VTK file
ts_surf = field.term_shock_surface()
ts_surf = translate(ts_surf, y_shift=-y_shift, z_shift=z_shift)
ts_surf = rotate_x(ts_surf, theta_attach, x_origin=1, y_origin=y_origin, 
    z_origin=z_plane)
ts_grid = pv.StructuredGrid(ts_surf[:,:,0], ts_surf[:,:,1], ts_surf[:,:,2])
ts_grid.save("terminating_shock.vtk")