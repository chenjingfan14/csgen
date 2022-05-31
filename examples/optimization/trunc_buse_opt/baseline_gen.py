"""
Generating a baseline truncated Busemann flow field contour for optimization.
Entrance flow conditions assume that flow has been compressed by a 6 degree
wedge forebody.

Author: Reece Otto 13/02/2022
"""
from csgen.busemann import busemann_M1_p3p1
from csgen.atmosphere import atmos_interp
from csgen.shock_relations import beta_oblique, M2_oblique, p2_p1_oblique, \
                                  T2_T1_oblique
from math import pi, tan
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# free-stream flow conditions
q0 = 50E3                                        # dynamic pressure [Pa]
M0 = 10                                          # Mach number
gamma = 1.4                                      # ratio of specific heats
p0 = 2 * q0 / (gamma * M0*M0)                    # static pressure [Pa]
T0 = atmos_interp(p0, 'Pressure', 'Temperature') # temperature [K]

# station 1 flow conditions (free-stream compressed by 6deg wedge)
alpha = 6 * pi / 180                       # forebody wedge angle [rad]
beta = beta_oblique(alpha, M0, gamma)      # forebody shock angle [rad]
M1 = M2_oblique(beta, M0, gamma)           # Mach number
p1 = p2_p1_oblique(beta, M0, gamma) * p0   # static pressure [Pa]
T1 = T2_T1_oblique(beta, M0, gamma) * T0   # temperature [K]

print('Busemann flow field entrance conditions:')
print(f'M1 = {M1:.4}')
print(f'p1 = {p1:.4} Pa')
print(f'T1 = {T1:.4} K \n')

# Busemann flow field design parameters
p3 = 50E3                     # desired exit pressure [Pa]
p3_p1 = p3/p1                 # desired compression ratio
dtheta = 0.001                # theta step size [rad]
delta = 5 * pi / 180          # truncation angle [rad]
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
scale = 1 / trunc_coords[-1][1]
z_shift = scale * abs(trunc_coords[-1][2])
trunc_coords = field.xyz_coords(scale=[1, scale, scale], 
    translate=[0, 0, z_shift], trunc_angle=delta)
stream_coords = field.xyz_coords(scale=[1, scale, scale], 
    translate=[0, 0, z_shift])

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

# create baseline solution directory
path = os.getcwd()
base_dir = path + '/solutions/trunc_buse_0/'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# create gas model input file
f = open('solutions/trunc_buse_0/ideal-air.inp', 'w')
f.write('model = "IdealGas" \nspecies = {\'air\'}')
f.close()

# create puffin script
puff_txt = f"""config.axisymmetric = True
L_ext = 0.25

init_gas_model('ideal-air-gas-model.lua')
gas1 = GasState(config.gmodel)
gas1.p = {p1} # Pa
gas1.T = {T1} # K
gas1.update_thermo_from_pT()
gas1.update_sound_speed()
M1 = {M1}
V1 = M1 * gas1.a

config.max_step_relax = 40

import csv
xs = []; ys = []
with open('trunc_contour_0.csv', 'r') as df:
    pt_data = csv.reader(df, delimiter=' ', skipinitialspace=False)
    for row in pt_data:
        if row[0] == '#': continue
        xs.append(float(row[0]))
        ys.append(float(row[1]))

x_thrt = xs[-1]
y_thrt = ys[-1]

from eilmer.spline import CubicSpline
busemann_contour = CubicSpline(xs, ys)
def upper_y(x):
    return busemann_contour(x) if x < x_thrt else y_thrt
def lower_y(x): return 0.0
def lower_bc(x): return 0
def upper_bc(x): return 0

config.max_x = x_thrt + L_ext
config.dx = config.max_x/1000

st1 = StreamTube(gas=gas1, velx=V1, vely=0.0,
                 y0=lower_y, y1=upper_y,
                 bc0=lower_bc, bc1=upper_bc,
                 ncells=75)
"""
f = open('solutions/trunc_buse_0/trunc_buse_0.py', 'w')
f.write(puff_txt)
f.close()

# save truncated contour as a CSV file
with open('solutions/trunc_buse_0/trunc_contour_0.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for i in range(len(trunc_coords)):
        writer.writerow([trunc_coords[len(trunc_coords)-1-i][2], 
            trunc_coords[len(trunc_coords)-1-i][1]])

# save bash bash script to run simulation
bash_txt = """#!/bin/bash
prep-gas ideal-air.inp ideal-air-gas-model.lua
puffin-prep --job=trunc_buse_0
puffin --job=trunc_buse_0
puffin-post --job=trunc_buse_0 --output=vtk
puffin-post --job=trunc_buse_0 --output=stream --cell-index=$ --stream-index=0
"""
f = open('solutions/trunc_buse_0/run-trunc_buse_0.sh', 'w')
f.write(bash_txt)
f.close()

print('\nSimulation files have been created for baseline contour.')