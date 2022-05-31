"""
Designing a conical waverider.

Author: Reece Otto 14/12/2021
"""
from csgen.conical_flow import conical_M0_thetac
from csgen.stream_utils import waverider_stream_trace
from csgen.grid import StructuredGrid
from csgen.waverider_utils import top_surface
from nurbskit.path import Ellipse, BSpline
from nurbskit.utils import auto_knot_vector
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import pi, sin, cos, tan

#------------------------------------------------------------------------------#
#                       Generating Conical Flow Field                          #
#------------------------------------------------------------------------------#
# generate conical flow field
cone_vals = {
    'mach_no': 8,
    'cone_angle': 10*pi/180,
    'gamma': 1.4,
    'length': 10,
    'beta_guess': 20*pi/180,
    'tol': 1.0E-6,
    'dtheta': 0.01*pi/180,
    'max_steps': 10000,
    'interp_sing': True,
    'print_freq': 50,
    'verbosity': 1 
}
field = conical_M0_thetac(cone_vals)

# generate canonical streamline
stream = field.Streamline(cone_vals)
stream = stream.scale(y_scale=-1)
field.plot(stream)

# generate cone and shock surfaces
z_base = 3.5
cone_surf = field.cone_surface(z_base)
shock_surf = field.shock_surface(z_base)

#------------------------------------------------------------------------------#
#                      Designing Waverider Base Shape                          #
#------------------------------------------------------------------------------#
# define waverider design parameters
n_phi = 101

# create cone cross-section at z=z_base
r_cone = z_base * tan(field.thetac)
cone_base = Ellipse(r_cone, r_cone)
cone_base_coords = cone_base.discretize(n_points=101)

# create shock cross-section at z=z_base
r_shock = z_base * tan(field.beta)
shock_base = Ellipse(r_shock, r_shock)
shock_base_coords = shock_base.discretize(n_points=101)

# create base contour for bottom surface of waverider
phi_int = -55*pi/180
x_shock = r_shock*cos(phi_int)
y_shock = r_shock*sin(phi_int)
ang_shock = pi/2 + phi_int
rd_param = 2*x_shock/100
del_x_rd = rd_param*cos(ang_shock)
del_y_rd = rd_param*sin(ang_shock)
max_y_bot = -1.01*r_cone
max_y_top = -0.7*r_cone

P_bot = [[0, max_y_bot],
         [x_shock/4, max_y_bot],
         [x_shock/2, max_y_bot*1.02],
         [x_shock - 2*del_x_rd, y_shock],
         [x_shock - del_x_rd, y_shock - del_y_rd],
         [x_shock, y_shock]]
p_bot = 3
U_bot = auto_knot_vector(len(P_bot), p_bot)
wr_bot = BSpline(P=P_bot, p=p_bot, U=U_bot)
wr_bot_coords = wr_bot.discretize(n_points=n_phi)

# create base contour for top surface of waverider
P_top = [[0, max_y_top],
         [x_shock/4, max_y_top],
         [x_shock/2, max_y_top*1.02],
         [x_shock - del_x_rd, y_shock + 2*del_y_rd],
         [x_shock + del_x_rd, y_shock + del_y_rd],
         [x_shock, y_shock]]
p_top = 3
U_top = auto_knot_vector(len(P_top), p_top)
wr_top = BSpline(P=P_top, p=p_top, U=U_top)
wr_top_coords = wr_top.discretize(n_points=n_phi)

# plot cross-section at z=z_base
plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
matplotlib.rc('text', usetex=True)
ax = plt.axes()
ax.plot(cone_base_coords[:,0], cone_base_coords[:,1], 'k', label='Cone Surface')
ax.plot(shock_base_coords[:,0], shock_base_coords[:,1], 'r', 
    label='Shock Surface')
ax.plot(wr_bot_coords[:,0], wr_bot_coords[:,1], 'b', label='Base Contour')
ax.plot(wr_top_coords[:,0], wr_top_coords[:,1], 'b')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

plt.axis('square')
plt.grid()
plt.legend(bbox_to_anchor=[1.0, 1.0])
plt.savefig('waverider_base.svg', bbox_inches='tight')

#------------------------------------------------------------------------------#
#                  Streamline-Tracing Waverider Surfaces                       #
#------------------------------------------------------------------------------#
# generate top and bottom surface grids
surf_vals = {
    'z_base': z_base,
    'n_phi': n_phi,
    'n_z': 50,
    'tol': 1.0E-5
}
bot_surf_right = waverider_stream_trace(surf_vals, wr_bot_coords, 
    stream.xyz_coords)
bot_surf_left = np.flip(bot_surf_right.copy()[1:], 0)
for i in range(len(bot_surf_left)):
    for j in range(len(bot_surf_left[0])):
        bot_surf_left[i][j][0] *= -1
bot_surf = np.concatenate((bot_surf_left, bot_surf_right))

wr_top_left_coords = wr_top_coords.copy()[1:][::-1]
for i in range(len(wr_top_left_coords)):
    wr_top_left_coords[i][0] *=-1
wr_top_coords = np.concatenate((wr_top_left_coords, wr_top_coords))
top_surf = top_surface(wr_top_coords, bot_surf)

# generate VTK surfaces
StructuredGrid(bot_surf).export_to_vtk_xml(file_name='waverider_bot')
StructuredGrid(top_surf).export_to_vtk_xml(file_name='waverider_top')
StructuredGrid(cone_surf).export_to_vtk_xml(file_name='cone_surf')
StructuredGrid(shock_surf).export_to_vtk_xml(file_name='shock_surf')