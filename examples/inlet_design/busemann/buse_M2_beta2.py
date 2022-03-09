"""
Generating a Busemann flow field by using M2 and beta as design parameters.

Author: Reece Otto 11/02/2022
"""
from csgen.busemann import busemann_M2_beta2
from math import pi
import matplotlib.pyplot as plt
import numpy as np

# Busemann flow field design parameters
M2 = 3                        # Mach number at station 2
beta2 = 30*pi/180             # angle of terminating shock
gamma = 1.4                   # ratio of specific heats
dtheta = 0.01 * pi / 180      # theta step size [rad]
field = busemann_M2_beta2(M2, beta2, gamma, dtheta)

# create plot of streamline
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
ax = plt.axes()
stream_coords = field.xyz_coords()
ax.plot(stream_coords[:,2], stream_coords[:,1], 'k-', label='Busemann Contour')
axis_coords = np.array([[np.amin(stream_coords[:,2]), 0],
	                    [np.amax(stream_coords[:,2]), 0]])
ax.plot(axis_coords[:,0], axis_coords[:,1], 'k-.', label='Axis of Symmetry')
mw_coords = np.array([[stream_coords[:,2][-1], stream_coords[:,1][-1]],
	                  [0, 0]])
ax.plot(mw_coords[:,0], mw_coords[:,1], 'r--', label='Entrance Mach Wave')
ts_coords = np.array([[0, 0],
	                  [stream_coords[:,2][0], stream_coords[:,1][0]]])
ax.plot(ts_coords[:,0], ts_coords[:,1], 'r-', label='Terminating Shock Wave')
ax.set_xlabel('$z$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()