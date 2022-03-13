"""
Generating a Busemann flow field by using M2 and beta as design parameters.

Author: Reece Otto 11/02/2022
"""
from csgen.busemann import busemann_M2_beta2
from math import pi
import matplotlib.pyplot as plt
import numpy as np

# Busemann flow field design parameters
M2 = 3                # Mach number at station 2
beta2 = 30 * pi/180   # angle of terminating shock
gamma = 1.4           # ratio of specific heats
dtheta = 0.05 * pi/180 # theta step size [rad]
field = busemann_M2_beta2(M2, beta2, gamma, dtheta)

# create plot of streamline
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 20
    })
ax = plt.axes()
stream = field.streamline()
ax.plot(stream.zs, stream.ys, 'k-', label='Busemann Contour')
ax.plot([stream.zs[0], stream.zs[-1]], [0, 0], 'k-.', label='Axis of Symmetry')
ax.plot([stream.zs[0], 0], [stream.ys[0], 0], 'r--', label='Entrance Mach Wave')
ax.plot([0, stream.zs[-1]], [0, stream.ys[-1]], 'r-', 
    label='Terminating Shock Wave')
ax.set_xlabel('$z$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()
fig.savefig('buse_M2_beta2.svg', bbox_inches='tight')