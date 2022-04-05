"""
Designing a conical waverider.

Author: Reece Otto 14/12/2021
"""
import os
from csgen.atmosphere import atmos_interp
from csgen.conical_field import conical_M0_beta
from csgen.stream_utils import waverider_stream_trace
from csgen.waverider_utils import top_surface
from nurbskit.path import BSpline, Ellipse
from nurbskit.utils import auto_knot_vector
from math import pi, tan, cos, sin, atan2, sqrt, atan
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import csv

#------------------------------------------------------------------------------#
#                    Define Design Values and Settings                         #
#------------------------------------------------------------------------------#
# design values for conical field and streamline
cone_vals = {
    'M0': 12,            # free-stream Mach number
    'beta': 10.5*pi/180, # shock angle [rad]
    'gamma': 1.4,        # ratio of specific heats
    'L_field': 10,       # length of conical field when tracing streamline
    'r0': 1              # initial radius of streamline
}

# integration settings
settings = {
    'dtheta': 0.01*pi/180, # integration step size for theta [rad]
    'max_steps': 10000,    # maximum number of integration steps
    'print_freq': 20,      # printing frequency of integration info
    'verbosity': 1         # verbosity level
}

# settings for streamline tracing
wr_vals = {
	'z_base': 5,     # z location of waverider base
	'n_phi': 51,     # number of points in phi direction
	'n_z': 51,       # number of points in z direction
	'tol': 1.0E-4,   # tolerance for streamline-finding algorithm
	'save_VTK': True # option to save surface as VTK file
}

# establish working directory
main_dir = os.getcwd()
working_dir = main_dir + '/waverider'
if not os.path.exists(working_dir):
    os.makedirs(working_dir)
os.chdir(working_dir)
#------------------------------------------------------------------------------#
#                         Generate Conical Flow Field                          #
#------------------------------------------------------------------------------#
# generate flow field and streamline
field = conical_M0_beta(cone_vals, settings)
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
max_y = 1.01 * np.amin(cone_base_coords[:,1])
phi_intercept = 60 * pi / 180
max_x = r_shock * cos(phi_intercept)
min_y = -r_shock * sin(phi_intercept)
"""
P = [[-max_x, min_y], [-max_x/2, (max_y + min_y)/1.9], [-max_x/3, max_y], 
     [0, max_y], 
     [max_x/3, max_y], [max_x/2,(max_y + min_y)/1.9], [max_x, min_y]]
"""
P = [[-max_x, min_y], [-max_x/2, max_y], [0, max_y], [max_x/2,max_y], 
     [max_x, min_y]]
p = 3
U = auto_knot_vector(len(P), p)
wr_base = BSpline(P=P, p=p, U=U)
wr_base_coords = wr_base.list_eval(n_points=n_phi)

# create baseline contour for top surface of waverider
P = [[-max_x, min_y], [-max_x/2, max_y*0.9], [0, max_y*0.8], 
     [max_x/2,max_y*0.9], [max_x, min_y]]
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
ax = plt.axes()
ax.plot(wr_base_coords[:,0], wr_base_coords[:,1], 'b', label='Waverider Base')
ax.plot(wr_top_coords[:,0], wr_top_coords[:,1], 'b')
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
wr_coords = waverider_stream_trace(wr_vals)
top_surface = top_surface(wr_top_coords, wr_coords)

#------------------------------------------------------------------------------#
#              Evaluate pressure field across base cross-section               #
#------------------------------------------------------------------------------#
# calculate remaining free-stream properties from US standard atmopshere 1976
q0 = 50E3   # dynamic pressure (Pa)
M0 = cone_vals['M0']
gamma = cone_vals['gamma']
p0 = 2 * q0 / (gamma * M0*M0)                    # static pressure (Pa)
T0 = atmos_interp(p0, 'Pressure', 'Temperature') # temperature (K)
a0 = atmos_interp(p0, 'Pressure', 'Sonic Speed') # sonic speed (m/s)
V0 = M0 * a0                                     # flight speed (m/s)

# actual shock wave created by waverider
wrs_coords = np.nan * np.ones(wr_coords.shape)
n_z = wr_vals['n_z']
for i in range(n_phi):
    for j in range(n_z):
        x_ij = wr_coords[i,j,0]
        z_ij = wr_coords[i,j,2]
        wrs_coords[i][j][0] = x_ij
        wrs_coords[i][j][1] = -sqrt((tan(field.beta) * z_ij)**2 - x_ij**2)
        wrs_coords[i][j][2] = z_ij

# save all surfaces as VTK files
wrs_grid = pv.StructuredGrid(wrs_coords[:,:,0], wrs_coords[:,:,1], 
    wrs_coords[:,:,2])
wrs_grid.save("waverider_shock.vtk")

# create grid between waverider bottom surface and shock
n_y_points = 20
exit_mesh = np.nan * np.ones((n_phi, n_y_points, 3))
for i in range(n_phi):
    y_shock = wrs_coords[i,-1,1]
    dy = (wr_coords[i,-1,1] - y_shock) / (n_y_points - 1)
    for j in range(n_y_points):
        exit_mesh[i][j][0] = wr_coords[i,-1,0]
        exit_mesh[i][j][1] = y_shock + j * dy
        exit_mesh[i][j][2] = z_base
exit_grid = pv.StructuredGrid(exit_mesh[:,:,0], exit_mesh[:,:,1], 
    exit_mesh[:,:,2])
exit_grid.save("exit_grid.vtk")

# evaluate pressure at each grid point
exit_theta = np.nan * np.ones((n_phi, n_y_points))
exit_delta = np.nan * np.ones((n_phi, n_y_points))
exit_mach = np.nan * np.ones((n_phi, n_y_points))
exit_pressure = np.nan * np.ones((n_phi, n_y_points))
exit_temp = np.nan * np.ones((n_phi, n_y_points))
exit_theta = np.nan * np.ones((n_phi, n_y_points))
print('\nEvaluating flow field at exit plane.\n')
for i in range(n_phi):
    for j in range(n_y_points):
        x_ij = exit_mesh[i][j][0]
        y_ij = exit_mesh[i][j][1]
        z_ij = exit_mesh[i][j][2]

        # calculate theta but make slightly lower to ensure in valid range
        theta_ij = atan(sqrt(x_ij**2 + y_ij**2) / z_ij) - 1E-6
        exit_theta[i][j] = theta_ij
        exit_mach[i][j] = field.M(theta_ij)
        exit_delta[i][j] = atan(field.v(theta_ij) / field.u(theta_ij))
        exit_pressure[i][j] = field.p(theta_ij, p0)
        exit_temp[i][j] = field.T(theta_ij, T0)

max_p_ind = np.argmax(exit_pressure)
theta_attach = exit_theta.flatten()[max_p_ind]

# calculate average properties over plane
y_attach = np.amax(exit_mesh[:,:,1])

outflow = {
    'M': np.average(exit_mach),
    'p': np.average(exit_pressure),
    'T': np.average(exit_temp)
}

print(f'Inlet attachment coords: [0.0, {y_attach}, {z_base}]')
print(f'Inlet attachment angle: {theta_attach*180/pi:.4} deg')
print(f"""Average flow properties over exit plane:
Theta = {np.average(exit_theta)*180/pi:.4} deg
Flow angle = {np.average(exit_delta)*180/pi:.4} deg
Mach number = {np.average(exit_mach):.4}
Pressure = {np.average(exit_pressure):.4} Pa
Temperature = {np.average(exit_temp):.4} K
""")

# plot pressure field at z=z_base
plt.figure(figsize=(16, 9))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 20
    })
ax = plt.axes()
plt.grid()
ax.plot(wr_coords[:,-1,0], wr_coords[:,-1,1], 'k')
ax.plot(wrs_coords[:,-1,0], wrs_coords[:,-1,1], 'k')

from csgen.waverider_utils import normal_exit_shock
norm_shock_ys = np.nan * np.ones(len(wrs_coords[:,-1,0]))
for i in range(len(norm_shock_ys)):
    x = wrs_coords[i,-1,0]
    norm_shock_ys[i] = normal_exit_shock(x, field.beta, theta_attach, y_attach, 
        z_base)

ax.plot(wrs_coords[:,-1,0], norm_shock_ys, 'r--')
plt.contourf(exit_mesh[:,:,0], exit_mesh[:,:,1], exit_pressure, 100)
cbar = plt.colorbar()
cbar.set_label('Pressure (Pa)', rotation=90)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.savefig('exit_pressure.svg', bbox_inches='tight')

# save exit cross-section as CSV file
with open('exit_shape.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for i in range(len(wr_coords)):
        writer.writerow([wr_coords[i,-1,0], wr_coords[i,-1,1]])
    for j in range(1, len(wrs_coords)):
        writer.writerow([wrs_coords[len(wrs_coords)-j-1,-1,0], 
            wrs_coords[len(wrs_coords)-j-1,-1,1]])

# create directory for diffuser simulation
os.chdir(main_dir)
diffuser_dir = main_dir + '/diffuser'
if not os.path.exists(diffuser_dir):
    os.makedirs(diffuser_dir)
os.chdir(diffuser_dir)
import json
with open('inflow.json', 'w') as f:
  json.dump(outflow, f, ensure_ascii=False, indent=2)