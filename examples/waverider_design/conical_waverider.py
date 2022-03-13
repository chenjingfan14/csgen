"""
Designing a conical waverider.

Author: Reece Otto 14/12/2021
"""
from csgen.atmosphere import atmos_interp
from csgen.conical_field import conical_M0_thetac
from csgen.isentropic_flow import p_pt
from csgen.stream_utils import waverider_stream_trace
from nurbskit.path import Ellipse, BSpline
from nurbskit.utils import auto_knot_vector
import matplotlib.pyplot as plt
import numpy as np
from math import pi, tan, cos, sin, atan2, sqrt, atan
import pyvista as pv
import csv

#------------------------------------------------------------------------------#
#                         Waverider design parameters                          #
#------------------------------------------------------------------------------#
# free-stream flow parameters
M0 = 10     # free-stream Mach number
q0 = 50E3   # dynamic pressure (Pa)
gamma = 1.4 # ratio of specific heats

# geometric properties of conical flow field
dtheta = 0.01 * pi / 180 # integration step size (rad)
thetac = 7 * pi / 180    # angle of conical shock (rad)

# geometric properties of waverider
z_base = 5     # z plane where base of waverider exists
n_streams = 51 # number of streamlines to be traced around base shape
n_z = 51       # number of points evaluated in z direction

#------------------------------------------------------------------------------#
#                       Generating conical flow field                          #
#------------------------------------------------------------------------------#
# calculate remaining free-stream properties from US standard atmopshere 1976
p0 = 2 * q0 / (gamma * M0*M0)                    # static pressure (Pa)
T0 = atmos_interp(p0, 'Pressure', 'Temperature') # temperature (K)
a0 = atmos_interp(p0, 'Pressure', 'Sonic Speed') # sonic speed (m/s)
V0 = M0 * a0                                     # flight speed (m/s)

# generate flow field
field = conical_M0_thetac(M0, thetac, gamma, dtheta, 
    beta_guess=20*pi/180, n_steps=10000, max_iter=10000, tol=1E-6, 
    interp_sing=True, verbosity=1, print_freq=10)
stream_coords = field.streamline(scale=[1, -1, 1], L_field=2*z_base)

# create plot of streamline
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 20
    })
ax = plt.axes()
max_z = np.amax(stream_coords[:,2])
axis_coords = np.array([[0, 0],
	                    [max_z, 0]])
shock_coords = np.array([[0, 0],
	                     [max_z, -max_z * tan(field.beta)]])
cone_coords = np.array([[0, 0],
	                     [max_z, -max_z * tan(field.thetac)]])

ax.plot(stream_coords[:,2], stream_coords[:,1], 'b-', label='Streamline')
ax.plot(axis_coords[:,0], axis_coords[:,1], 'k-.', label='Axis of Symmetry')
ax.plot(shock_coords[:,0], shock_coords[:,1], 'r-', label='Shockwave')
ax.plot(cone_coords[:,0], cone_coords[:,1], 'k-', label='Cone')

ax.set_xlabel('$z$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()
fig.savefig('conical_field.svg', bbox_inches='tight')

#------------------------------------------------------------------------------#
#                      Generating waverider base shape                         #
#------------------------------------------------------------------------------#
# create cone cross-section at z=z_base
r_cone = z_base * tan(field.thetac)
cone_base = Ellipse(r_cone, r_cone)
cone_base_coords = cone_base.list_eval(n_points=n_streams)

# create shock cross-section at z=z_base
r_shock = z_base * tan(field.beta)
shock_base = Ellipse(r_shock, r_shock)
shock_base_coords = shock_base.list_eval(n_points=n_streams)

# create baseline contour for bottom surface of waverider
max_y = 1.05 * np.amin(cone_base_coords[:,1])
phi_intercept = 50 * pi / 180
max_x = r_shock * cos(phi_intercept)
min_y = -r_shock * sin(phi_intercept)
P = [[-max_x, min_y], [-max_x/2, (max_y + min_y)/1.9], [-max_x/3, max_y], 
     [0, max_y], 
     [max_x/3, max_y], [max_x/2,(max_y + min_y)/1.9], [max_x, min_y]]
p = 3
U = auto_knot_vector(len(P), p)
wr_base = BSpline(P=P, p=p, U=U)
wr_base_coords = wr_base.list_eval(n_points=n_streams)

# plot cross-section at z=z_base
plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
ax = plt.axes()
ax.plot(wr_base_coords[:,0], wr_base_coords[:,1], 'b', label='Waverider Base')
ax.plot(cone_base_coords[:,0], cone_base_coords[:,1], 'r', 
	label='Conical Shock')
ax.plot(shock_base_coords[:,0], shock_base_coords[:,1], 'k', label='Base Cone')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.grid()
plt.legend()
plt.savefig('waverider_base.svg', bbox_inches='tight')
plt.show()

#------------------------------------------------------------------------------#
#                  Streamline-tracing waverider base shape                     #
#------------------------------------------------------------------------------#
# run streamline tracer
wr_coords = waverider_stream_trace(wr_base_coords, stream_coords, z_base, n_z, 
	tol=1E-6)

# generate shockwave cone surfaces
def cone_x(r, phi):
    return r * np.cos(phi)
    
def cone_y(r, phi):
    return r * np.sin(phi)
    
def cone_z(x, y, theta, sign):
    a = tan(theta)
    return sign * np.sqrt((x*x + y*y) / (a*a))

# conical shock surface
rs_cs = np.linspace(0, 11 * tan(field.beta), n_streams)
phis_cs = np.linspace(0, 2 * pi, n_streams)
Rs_cs, Phis_cs = np.meshgrid(rs_cs, phis_cs)
X_cs = cone_x(Rs_cs, Phis_cs)
Y_cs = cone_y(Rs_cs, Phis_cs)
Z_cs = cone_z(X_cs, Y_cs, field.beta, 1)

# imaginary cone surface
rs_c = np.linspace(0, 11 * tan(field.thetac), n_streams)
phis_c = np.linspace(0, 2 * pi, n_streams)
Rs_c, Phis_c = np.meshgrid(rs_c, phis_c)
X_c = cone_x(Rs_c, Phis_c)
Y_c = cone_y(Rs_c, Phis_c)
Z_c = cone_z(X_c, Y_c, field.thetac, 1)

# actual shock wave created by waverider
wrs_coords = np.nan * np.ones(wr_coords.shape)
for i in range(n_streams):
    for j in range(n_z):
        x_ij = wr_coords[i,j,0]
        z_ij = wr_coords[i,j,2]
        wrs_coords[i][j][0] = x_ij
        wrs_coords[i][j][1] = -sqrt((tan(field.beta) * z_ij)**2 - x_ij**2)
        wrs_coords[i][j][2] = z_ij

# save all surfaces as VTK files
wr_grid = pv.StructuredGrid(wr_coords[:,:,0], wr_coords[:,:,1], 
	wr_coords[:,:,2])
wr_grid.save("waverider.vtk")
cs_grid = pv.StructuredGrid(X_cs, Y_cs, Z_cs)
cs_grid.save("field_shock.vtk")
c_grid = pv.StructuredGrid(X_c, Y_c, Z_c)
c_grid.save("field_cone.vtk")
wrs_grid = pv.StructuredGrid(wrs_coords[:,:,0], wrs_coords[:,:,1], 
    wrs_coords[:,:,2])
wrs_grid.save("waverider_shock.vtk")

#------------------------------------------------------------------------------#
#              Evaluate pressure field across base cross-section               #
#------------------------------------------------------------------------------#
# create grid between waverider bottom surface and shock
n_y_points = 20
exit_mesh = np.nan * np.ones((n_streams, n_y_points, 3))
for i in range(n_streams):
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
exit_theta = np.nan * np.ones((n_streams, n_y_points))
exit_delta = np.nan * np.ones((n_streams, n_y_points))
exit_mach = np.nan * np.ones((n_streams, n_y_points))
exit_pressure = np.nan * np.ones((n_streams, n_y_points))
exit_temp = np.nan * np.ones((n_streams, n_y_points))
print('\nEvaluating flow field at exit plane.\n')
for i in range(n_streams):
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

# calculate average properties over plane
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
plt.contourf(exit_mesh[:,:,0], exit_mesh[:,:,1], exit_pressure, 100)
cbar = plt.colorbar()
cbar.set_label('Pressure (Pa)', rotation=90)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.savefig('exit_pressure.svg', bbox_inches='tight')
plt.show()

# save exit cross-section as CSV file
with open('exit_shape.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for i in range(len(wr_coords)):
        writer.writerow([wr_coords[i,-1,0], wr_coords[i,-1,1]])
    for j in range(1, len(wrs_coords)):
        writer.writerow([wrs_coords[len(wrs_coords)-j-1,-1,0], 
            wrs_coords[len(wrs_coords)-j-1,-1,1]])