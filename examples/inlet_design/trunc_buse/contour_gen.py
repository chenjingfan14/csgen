"""
Generating a truncated Busemann flow field contour.

Author: Reece Otto 13/02/2022
"""
from csgen.busemann import busemann_M1_p3p1
from math import pi, tan
import matplotlib.pyplot as plt
import numpy as np
import csv

# Busemann flow field design parameters
p1 = 2612.39371900317  # entrance static pressure [Pa]
M1 = 7.950350403130542 # entrance Mach number
T1 = 328.5921740025843 # entrance temperature [K]
gamma = 1.4            # ratio of specific heats
p3 = 50E3              # desired exit pressure [Pa]
p3_p1 = p3/p1          # desired compression ratio
dtheta = 0.001         # theta step size [rad]
delta = 7 * pi / 180   # truncation angle [rad]
field = busemann_M1_p3p1(M1, p3_p1, gamma, dtheta)

# create plot of streamline
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
ax = plt.axes()

trunc_coords = field.xyz_coords(trunc_angle=delta)
z_shift = abs(trunc_coords[-1][2])
trunc_coords = field.xyz_coords(translate=[0, 0, z_shift], trunc_angle=delta)
stream_coords = field.xyz_coords(translate=[0, 0, z_shift])

ax.plot(stream_coords[:,2], stream_coords[:,1], 'k-', label='Busemann Contour')
ax.plot(trunc_coords[:,2], trunc_coords[:,1], 'b-', label='Truncated Contour')

def trunc_line(delta, z):
    return tan(pi - field.mu - delta) * z + trunc_coords[-1][1]

min_z = trunc_coords[-1][2]
tl_coords = np.array([[min_z, trunc_line(delta, min_z)],
                      [z_shift, 0]])
ax.plot(tl_coords[:,0], tl_coords[:,1], 'g--', label='Truncation Line')
axis_coords = np.array([[np.amin(stream_coords[:,2]), 0],
	                [np.amax(stream_coords[:,2]), 0]])
ax.plot(axis_coords[:,0], axis_coords[:,1], 'k-.', label='Axis of Symmetry')
mw_coords = np.array([[stream_coords[:,2][-1], stream_coords[:,1][-1]],
                      [z_shift, 0]])
ax.plot(mw_coords[:,0], mw_coords[:,1], 'r--', label='Entrance Mach Wave')
ax.set_xlabel('$z$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()

# save truncated contour as a CSV file
with open('trunc_contour.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for i in range(len(trunc_coords)):
        writer.writerow([trunc_coords[len(trunc_coords)-1-i][2], 
            trunc_coords[len(trunc_coords)-1-i][1]])
