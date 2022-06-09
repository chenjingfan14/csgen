from nurbskit.path import BSpline
from nurbskit.surface import BSplineSurface
from nurbskit.utils import auto_knot_vector
from nurbskit.visualisation import path_plot_2D
from nurbskit.file_io import nurbs_surf_to_iges
from csgen.grid import StructuredGrid
from math import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt

# feature-based parameters
L1 = 3.0; L2 = 1.0; W = 1.0; H1 = 0.42; H2 = 0.18; r = 0.01 
theta1 = 16*pi/180; theta2 = 5.7*pi/180
L = L1 + L2

# control point parameters
x0 = 0.0; y0 = H1
x1 = W/5; y1 = H1
x2 = W/2; y2 = H1/2
x3 = 3*W/4; y3 = 5*r
x4 = W - r; y4 = r
x5 = W; y5 = r
x6 = W; y6 = -r
x7 = W - r; y7 = -r
x8 = 3*W/4; y8 = -3*H2/4
x9 = W/5; y9 = -H2
x10 = 0.0; y10 = -H2

# left-half of control net at base
P_right = [[x0, y0, L],
           [x1, y1, L],
           [x2, y2, L],
           [x3, y3, L],
           [x4, y4, L],
           [x5, y5, L],
           [x6, y6, L],
           [x7, y7, L],
           [x8, y8, L],
           [x9, y9, L],
           [x10, y10, L]]

# full control net at base
P_base = np.zeros((2*len(P_right)-1, 3))
P_base[:len(P_right)] = P_right           # right half of contour
P_base[len(P_right):] = P_right[::-1][1:] # left half of contour
P_base[len(P_right):][:,0] *= -1          # negate x vals on left half

# create B-Spline curve for base contour
q = 3
n_phi = 200
V = auto_knot_vector(len(P_base), q)
base = BSpline(P=P_base, p=q, U=V)

# plot base contour
plt.figure(figsize=(16, 9))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 20
    })
ax = path_plot_2D(base, show_control_net=False)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.grid()
plt.savefig('base.png', bbox_inches='tight')

# construct B-Spline surface
N_u = 5
P = np.zeros((N_u, len(P_base), 3))

# create blunted nose with radius = r
fan_angle = 2*pi/(len(P_base) - 1)
for i in range(len(P_base)):
    P[1][i] = [4*r*cos(pi/2 - i*fan_angle), 4*r*sin(pi/2 - i*fan_angle), 0.0]

# decrease cross-sectional area towards nose
for i in range(2, N_u):
    P[i] = i/(N_u - 1) * P_base

    # ensure edges have constant radius
    # right-side edge
    P[i][4][0] = P[i][5][0] - r
    P[i][5][1] = r
    P[i][6][1] = -r
    P[i][7][0] = P[i][6][0] - r

    # left-side edge
    P[i][16][0] = P[i][15][0] + r
    P[i][15][1] = r
    P[i][14][1] = -r
    P[i][13][0] = P[i][14][0] + r
p = 3
U = auto_knot_vector(len(P), p)
vehicle = BSplineSurface(p=p, q=q, U=U, V=V, P=P)

# create frustrum with NURBS
h1 = 0.175; h2 = 0.45; l = 2.0

# export as IGES and VTK
nurbs_surf_to_iges(vehicle.cast_to_nurbs_surface(), file_name='HTV-2')
vehicle_grid = vehicle.discretize(N_u=100, N_v=101)
StructuredGrid(vehicle_grid).export_to_vtk_xml(file_name='HTV-2')