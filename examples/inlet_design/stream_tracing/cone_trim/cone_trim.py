"""
Streamline tracing a waverider exit shape through a Busemann flow field.
Trimming the inlet by the Busemann shock as well as the waverider shock.

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


def waverider_shock_2D(phi, z, beta, y_shift, z_shift, y_rot, z_rot, alpha):
    factor1 = cos(alpha) * \
              sqrt(((tan(beta))**2 - (cos(phi))**2)/((cos(phi))**2 + 1))
    factor2 = z_shift + z_rot + (y - y_shift - y_rot) * tan(alpha) + \
              (z - z_rot) / cos(alpha)
    factor3 = 

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