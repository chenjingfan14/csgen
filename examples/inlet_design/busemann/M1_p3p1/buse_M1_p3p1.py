"""
Generating a Busemann flow field by using M1 and p3/p1 as design parameters.

Author: Reece Otto 11/02/2022
"""
from csgen.busemann import busemann_M1_p3p1
from math import pi
import matplotlib.pyplot as plt
import numpy as np

# Busemann flow field design parameters
q1 = 50E3                     # entrance dynamic pressure [Pa]
M1 = 10                       # entrance Mach number
gamma = 1.4                   # ratio of specific heats
p1 = 2 * q1 / (gamma * M1*M1) # entrance static pressure [Pa]
p3 = 50E3                     # desired exit pressure [Pa]
p3_p1 = p3/p1                 # desired compression ratio
dtheta = 0.05 * pi/180        # theta step size [rad]
field = busemann_M1_p3p1(M1, p3_p1, gamma, dtheta)

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
fig.savefig('buse_M1_p3p1.svg', bbox_inches='tight')