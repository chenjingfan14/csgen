"""
Generating a conical flow field characterised by M0 and beta.

Author: Reece Otto 14/12/2021
"""
from csgen.atmosphere import atmos_interp
from csgen.conical_field import conical_M0_beta
from csgen.isentropic_flow import p_pt
import matplotlib.pyplot as plt
import numpy as np
from math import pi, tan

# free-stream flow parameters
M0 = 10                       # free-stream Mach number
q0 = 50E3                     # dynamic pressure (Pa)
gamma = 1.4                   # ratio of specific heats
p0 = 2 * q0 / (gamma * M0*M0) # static pressure (Pa)

# calculate remaining free-stream properties from US standard atmopshere 1976
T0 = atmos_interp(p0, 'Pressure', 'Temperature')
a0 = atmos_interp(p0, 'Pressure', 'Sonic Speed')
V0 = M0 * a0

# calculate free-stream stagnation properties


# geometric properties of conical flow field
dtheta = 0.01*pi / 180 # integration step size (rad)
beta = 15 * pi / 180   # angle of conical shock (rad)

# generate flow field
field = conical_M0_beta(M0, beta, gamma, dtheta=dtheta)
stream_coords = field.streamline(L_field=5)

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
	                     [max_z, max_z * tan(field.beta)]])
cone_coords = np.array([[0, 0],
	                     [max_z, max_z * tan(field.thetac)]])

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
