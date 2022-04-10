"""
Designing a conical waverider.

Author: Reece Otto 14/12/2021
"""
import os
import json
from csgen.conical_field import conical_M0_beta
from csgen.atmosphere import atmos_interp

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
#                         Generate Conical Flow Field                          #
#------------------------------------------------------------------------------#
# import waverider design values
main_dir = os.getcwd()
waverider_dir = main_dir + '/waverider'
os.chdir(waverider_dir)
f = open('waverider_vals.json')
waverider_vals = json.load(f)
f.close()

# generate flow field and streamline, then plot
field = conical_M0_beta(waverider_vals['cone_vals'], waverider_vals['settings'])
Stream = field.Streamline(waverider_vals['cone_vals'], 
    waverider_vals['settings'])
Stream = Stream.scale(y_scale=-1)
field.plot(Stream)

# generate cone and shock surfaces
field.cone_surface(cone_vals['L_field'])
field.shock_surface(cone_vals['L_field'])

#------------------------------------------------------------------------------#
#               Define Cross-Sectional Shape of Waverider Base                 #
#------------------------------------------------------------------------------#
# TODO: is this worthy of its own python file?
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
# TODO: create mesh evaluation routine and call it here
# calculate remaining free-stream properties from US standard atmopshere 1976
q0 = 50E3   # dynamic pressure (Pa)
M0 = cone_vals['M0']
gamma = cone_vals['gamma']
p0 = 2 * q0 / (gamma * M0*M0)                    # static pressure (Pa)
T0 = atmos_interp(p0, 'Pressure', 'Temperature') # temperature (K)                                  # flight speed (m/s)

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

attach = {
    'y_attach': y_attach,
    'z_attach': z_base,
    'alpha_attach': theta_attach
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

# export inflow values for diffuser simulation
diffuser_dir = main_dir + '/diffuser'
os.chdir(diffuser_dir)
with open('inflow.json', 'w') as f:
  json.dump(outflow, f, ensure_ascii=False, indent=2)

# export attachment values for inlet design
inlet_dir = main_dir + '/inlet'
os.chdir(inlet_dir)
with open('attach.json', 'w') as f:
  json.dump(attach, f, ensure_ascii=False, indent=2)