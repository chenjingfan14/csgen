"""
Streamline tracing a waverider exit shape through a Busemann flow field.

Author: Reece Otto 07/03/2022
"""
from csgen.busemann import busemann_M1_p3p1
from csgen.stream_utils import busemann_stream_trace
from nurbskit.path import Ellipse
import pyvista as pv
from math import pi, sqrt
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
dtheta = 0.05 * pi/180 # theta step size [rad]
n_streams = 101        # number of streamlines

field = busemann_M1_p3p1(M1, p3_p1, gamma, dtheta, beta2_guess=0.2312, 
    M2_guess=5.45, print_freq=1000)

# create plot of streamline
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
ax = plt.axes()
stream = field.streamline()
ax.plot(stream.zs, stream.ys, 'k-', label='Busemann Contour')
ax.plot([stream.zs[0], stream.zs[-1]], [0, 0], 'k-.', label='Axis of Symmetry')
ax.plot([stream.zs[0], 0], [stream.ys[0], 0], 'r--', label='Entrance Mach Wave')
ax.plot([0, stream.zs[-1]], [0, stream.ys[-1]], 'r-', 
    label='Terminating Shock Wave')
ax.set_xlabel('$z$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()
fig.savefig('busemann.svg', bbox_inches='tight')

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
for i in range(len(cap_coords)):
    cap_coords[i][1] += abs(min_y) + 0.05*stream.ys[0]

# plot capture shape on entrance of flow field
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 20
    })
ax = plt.axes()
buse_ent = Ellipse(stream.ys[0], stream.ys[0])
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

# run streamline tracer and save the surface as a VTK file
inlet_coords = busemann_stream_trace(cap_coords, field)
inlet_grid = pv.StructuredGrid(inlet_coords[0], inlet_coords[1], 
        inlet_coords[2])
inlet_grid.save("waverider_inlet.vtk")

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