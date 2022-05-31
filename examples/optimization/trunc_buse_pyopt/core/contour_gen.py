"""
Generate truncated Busemann contour.

Author: Reece Otto 20/05/2022
"""
from csgen.buse_flow import busemann_M1_p3p1
from csgen.atmos import atmos_interp
from csgen.compress_flow import beta_oblique, M2_oblique, p2_p1_oblique, \
                                T2_T1_oblique
from csgen.file_io import coords_to_csv
from math import pi
import numpy as np
import csv
import json
import os

# calculate free-stream conditions
gamma = 1.4
M0 = 10
q0 = 50E3
p0 = 2*q0 / (gamma*M0*M0)
T0 = atmos_interp(p0, 'Pressure', 'Temperature')

# calculate conditions at inlet entrance
# note: assume free-stream is compressed by 6 deg wedge
theta = 6*pi/180
beta = beta_oblique(theta, M0, gamma)
M1 = M2_oblique(beta, theta, M0, gamma)
p1 = p2_p1_oblique(beta, M0, gamma) * p0
T1 = T2_T1_oblique(beta, M0, gamma) * T0

# export inflow conditions as json
inflow = {
    'mach_no': M1,
    'press': p1,
    'temp': T1
}
core_dir = os.getcwd()
main_dir = os.path.dirname(core_dir)
os.chdir(main_dir + '/sol_0')
with open('inflow.json', 'w') as write_file:
    json.dump(inflow, write_file)

# generate Busemann field
buse_vals = {
    'M1': 8.0,
    'p3_p1': 60E3/p1,
    'r0': 1.0,
    'gamma': gamma,
    'dtheta': 0.05*pi/180,
    'M2_guess': 5.3875,
    'beta2_guess': 0.2312,
    'max_steps': 10000,
    'print_freq': 500,
    'verbosity': 1
}
field = busemann_M1_p3p1(buse_vals)

# truncate and re-scale contour
field.Streamline = field.Streamline.truncate(trunc_angle=7*pi/180)
z_shift = abs(field.Streamline.zs[0])
field.Streamline = field.Streamline.translate(z_shift=z_shift)
scale_fac = 1.0 / field.Streamline.ys[0]
field.Streamline = field.Streamline.scale(scale_fac, scale_fac, scale_fac)

# save plot and csv of contour
field.plot(file_name='contour_0', show_mach_wave=False, show_exit_shock=False)
coords = np.zeros((len(field.Streamline.xyz_coords), 2))
coords[:,0] = field.Streamline.xyz_coords[:,2]
coords[:,1] = field.Streamline.xyz_coords[:,1]
coords_to_csv(coords, file_name=f'contour_0')