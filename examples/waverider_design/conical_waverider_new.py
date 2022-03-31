"""
Designing a conical waverider.

Author: Reece Otto 14/12/2021
"""
from csgen.conical_field import conical_M0_thetac
from csgen.stream_utils import waverider_stream_trace
from nurbskit.path import BSpline, Ellipse
from nurbskit.utils import auto_knot_vector
from math import pi, sin, cos, tan
import numpy as np
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------#
#                    Define Design Values and Settings                         #
#------------------------------------------------------------------------------#
# design values for conical field and streamline
cone_vals = {
    'M0': 10,           # free-stream Mach number
    'thetac': 7*pi/180, # angle of conical shock [rad]
    'gamma': 1.4,       # ratio of specific heats
    'L_field': 10,      # length of conical field when tracing streamline
    'r0': 1             # initial radius of streamline
}

# integration settings
settings = {
    'beta_guess': 20*pi/180, # initial guess for shock angle [rad]
    'tol': 1.0E-6,           # tolerance for thetac convergence [rad]
    'dtheta': 0.01*pi/180,   # integration step size for theta [rad]
    'max_steps': 10000,      # maximum number of integration steps
    'print_freq': 20,        # printing frequency of integration info
    'verbosity': 1           # verbosity level
}

# settings for streamline tracing
wr_vals = {
	'z_base': 5,     # z location of waverider base
	'n_phi': 51,     # number of points in phi direction
	'n_z': 51,       # number of points in z direction
	'tol': 1.0E-5,   # tolerance for streamline-finding algorithm
	'save_VTK': True # option to save surface as VTK file
}

#------------------------------------------------------------------------------#
#                         Generate Conical Flow Field                          #
#------------------------------------------------------------------------------#
# generate flow field and streamline
field = conical_M0_thetac(cone_vals, settings)
Stream = field.Streamline(cone_vals, settings)

# negate y-coords of streamline
Stream = Stream.scale(y_scale=-1)

# generate plot
field.plot(Stream)

# generate surfaces
field.cone_surface(cone_vals['L_field'])
field.shock_surface(cone_vals['L_field'])

#------------------------------------------------------------------------------#
#               Define Cross-Sectional Shape of Waverider Base                 #
#------------------------------------------------------------------------------#
# extract design information from settings dictionary
z_base = wr_vals['z_base']
n_phi = wr_vals['n_phi']

# create cone cross-section at z=z_base
r_cone = z_base * tan(field.thetac)
cone_base = Ellipse(r_cone, r_cone)
cone_base_coords = cone_base.list_eval(n_points=n_phi)

# create shock cross-section at z=z_base
r_shock = z_base * tan(field.beta)
shock_base = Ellipse(r_shock, r_shock)
shock_base_coords = shock_base.list_eval(n_points=n_phi)

# create baseline contour for bottom surface of waverider
max_y = 1.05 * np.amin(cone_base_coords[:,1])
phi_intercept = 55 * pi / 180
max_x = r_shock * cos(phi_intercept)
min_y = -r_shock * sin(phi_intercept)
P = [[-max_x, min_y], [-max_x/2, (max_y + min_y)/1.9], [-max_x/3, max_y], 
     [0, max_y], 
     [max_x/3, max_y], [max_x/2,(max_y + min_y)/1.9], [max_x, min_y]]
p = 3
U = auto_knot_vector(len(P), p)
wr_base = BSpline(P=P, p=p, U=U)
wr_base_coords = wr_base.list_eval(n_points=n_phi)

# plot cross-section at z=z_base
plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
ax = plt.axes()
ax.plot(wr_base_coords[:,0], wr_base_coords[:,1], 'b', label='Waverider Base')
ax.plot(cone_base_coords[:,0], cone_base_coords[:,1], 'k', 
	label='Base Cone')
ax.plot(shock_base_coords[:,0], shock_base_coords[:,1], 'r', label='Shockwave')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.grid()
plt.legend()
plt.savefig('waverider_base.svg', bbox_inches='tight')

# add base coords and conical streamline to wr_vals
wr_vals['base_coords'] = wr_base_coords
wr_vals['stream_coords'] = Stream.xyz_coords

#------------------------------------------------------------------------------#
#                            Run Streamline Tracer                             #
#------------------------------------------------------------------------------#
wr_surf = waverider_stream_trace(wr_vals)