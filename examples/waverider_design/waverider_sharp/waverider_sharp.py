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
    'length': 3.5,
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
cone_surf = field.cone_surface(cone_vals['length'])
shock_surf = field.shock_surface(cone_vals['length'])

#------------------------------------------------------------------------------#
#                      Designing Waverider Base Shape                          #
#------------------------------------------------------------------------------#
# define waverider design parameters
z_base = cone_vals['length']
n_phi = 51

# create cone cross-section at z=z_base
r_cone = z_base * tan(field.thetac)
cone_base = Ellipse(r_cone, r_cone)
cone_base_coords = cone_base.list_eval(n_points=n_phi)

# create shock cross-section at z=z_base
r_shock = z_base * tan(field.beta)
shock_base = Ellipse(r_shock, r_shock)
shock_base_coords = shock_base.list_eval(n_points=n_phi)

# create baseline contour for bottom surface of waverider
max_y = 1.01 * np.amin(cone_base_coords[:,1])
phi_intercept = 55 * pi / 180
max_x = r_shock * cos(phi_intercept)
min_y = -r_shock * sin(phi_intercept)
P = [[-max_x, min_y], [-max_x/2, -r_cone*1.07], [0, -r_cone*1.03], 
     [max_x/2, -r_cone*1.07], [max_x, min_y]]
p = 3
U = auto_knot_vector(len(P), p)
wr_bot = BSpline(P=P, p=p, U=U)
wr_bot_coords = wr_bot.list_eval(n_points=n_phi)

# create baseline contour for top surface of waverider
P = [[-max_x, min_y], [-max_x/2, max_y*0.8], [0, max_y*0.7], 
     [max_x/2,max_y*0.8], [max_x, min_y]]
p = 3
U = auto_knot_vector(len(P), p)
wr_top = BSpline(P=P, p=p, U=U)
wr_top_coords = wr_top.list_eval(n_points=n_phi)

# plot cross-section at z=z_base
plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
matplotlib.rc('text', usetex=True)
ax = plt.axes()
ax.plot(wr_bot_coords[:,0], wr_bot_coords[:,1], 'b', label='Base Contour')
ax.plot(wr_top_coords[:,0], wr_top_coords[:,1], 'b')
ax.plot(cone_base_coords[:,0], cone_base_coords[:,1], 'k', label='Cone Surface')
ax.plot(shock_base_coords[:,0], shock_base_coords[:,1], 'r', 
    label='Shock Surface')
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
    'z_base': cone_vals['length'],
    'n_phi': n_phi,
    'n_z': 50,
    'tol': 1.0E-4
}
bot_surf = waverider_stream_trace(surf_vals, wr_bot_coords, stream.xyz_coords)
top_surf = top_surface(wr_top_coords, bot_surf)

# generate VTK surfaces
StructuredGrid(bot_surf).export_to_vtk_xml(file_name='waverider_bot')
StructuredGrid(top_surf).export_to_vtk_xml(file_name='waverider_top')
StructuredGrid(cone_surf).export_to_vtk_xml(file_name='cone_surf')
StructuredGrid(shock_surf).export_to_vtk_xml(file_name='shock_surf')

#------------------------------------------------------------------------------#
#                          Generate NURBS Patches                              #
#------------------------------------------------------------------------------#
"""
Patch topology:
        _
       /|\
      / | \
     /  |  \
    / \ | / \
   /   \|/   \
  /     |     \
 /      |      \
/       |       \
-----------------
"""
# calculate points
nose = bot_surf[n_phi//2][0]
mid = bot_surf[n_phi//2][len(bot_surf[0])//2]
back = bot_surf[n_phi//2][-1]
back_left = bot_surf[0][0]
back_right = bot_surf[-1][0]
front_left = bot_surf[n_phi//8][0]
back_left = bot_surf[7*n_phi//8][0]

# construct boundaries
print(front_left)
print(mid)