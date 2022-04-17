"""
Designing a conical waverider.

Author: Reece Otto 14/12/2021
"""
import os
import json
from csgen.conical_field import conical_M0_beta
from csgen.stream_utils import waverider_stream_trace
from csgen.waverider_utils import top_surface, flow_field_2D, avg_props_2D
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
free_stream = waverider_vals['free_stream']
cone_vals = waverider_vals['cone_vals']
cone_settings = waverider_vals['cone_settings']
surf_vals = waverider_vals['surf_vals']

# generate flow field and streamline then save plot
field = conical_M0_beta(cone_vals, cone_settings)
Stream = field.Streamline(cone_vals, cone_settings)
Stream = Stream.scale(y_scale=-1)
field.plot(Stream)

# generate cone and shock surfaces
field.cone_surface(cone_vals['field_len'])
field.shock_surface(cone_vals['field_len'])

#------------------------------------------------------------------------------#
#                         Define Waverider Base Contour                        #
#------------------------------------------------------------------------------#
# extract design information from settings dictionary
z_base = surf_vals['z_base']
n_phi = surf_vals['n_phi']

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

#------------------------------------------------------------------------------#
#                            Run Streamline Tracer                             #
#------------------------------------------------------------------------------#
print('Running streamline tracer...')
wr_coords = waverider_stream_trace(surf_vals, wr_base_coords, Stream.xyz_coords)
print('Done.\n')
top_surface = top_surface(wr_top_coords, wr_coords)
wr_trim = wr_coords[1:-1,:,:]
wr_trim_grid = pv.StructuredGrid(wr_trim[:,:,0], wr_trim[:,:,1], wr_trim[:,:,2])
wr_trim_grid.save("wr_trim.vtk")

#------------------------------------------------------------------------------#
#                             Evaluate Exit Flow                               #
#------------------------------------------------------------------------------#
# actual shock wave created by waverider
wrs_coords = np.nan * np.ones(wr_coords.shape)
n_z = surf_vals['n_z']
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

# evaluate flow field over exit mesh
exit_data = flow_field_2D(field, exit_mesh, free_stream)
avg_theta, avg_delta, avg_M, avg_p, avg_T = avg_props_2D(exit_data)
outflow = {
    'mach_no': avg_M,
    'press': avg_p,
    'temp': avg_T
}
print('Average flow properties across exit plane:')
print(f'Flow angle: {avg_delta*180/pi:.4} deg')
print(f'Mach number: {avg_M:.4}')
print(f'Pressure: {avg_p:.4} Pa')
print(f'Temprature: {avg_T:.4} K')

# calculate inlet attachment data
y_attach = exit_data[:,:,1][n_phi//2][-1]
z_attach = exit_data[:,:,2][n_phi//2][-1]
theta_attach = exit_data[:,:,3][n_phi//2][-1]
attach = {
    'y_attach': y_attach,
    'z_attach': z_attach,
    'attach_angle': theta_attach
}

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
plt.contourf(exit_mesh[:,:,0], exit_mesh[:,:,1], exit_data[:,:,6], 100)
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
f = open('diffuser_vals.json')
diffuser_vals = json.load(f)
f.close()
diffuser_vals['inflow_data'] = outflow
with open('diffuser_vals.json', 'w') as f:
  json.dump(diffuser_vals, f, ensure_ascii=False, indent=2)

# export attachment values for inlet design
inlet_dir = main_dir + '/inlet'
os.chdir(inlet_dir)
with open('attach.json', 'w') as f:
  json.dump(attach, f, ensure_ascii=False, indent=2)