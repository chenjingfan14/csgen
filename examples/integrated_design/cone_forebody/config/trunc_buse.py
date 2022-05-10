"""
Generate truncated Busemann contour based on exit flow conditions from forebody.

Author: Reece Otto 29/03/2022
"""
from csgen.buse_flow import busemann_M1_p3p1
from math import pi, sqrt
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import csv

# print title
title_width = 38
print('-'*title_width)
print(f"{'Generating Truncated Busemann Diffuser':^{title_width}}")
print('-'*title_width)

# read inlet design parameters
config_dir = os.getcwd()
main_dir = os.path.dirname(config_dir)
f = open('design_vals.json')
design_vals = json.load(f)
f.close()
inlet_vals = design_vals['inlet']
free_stream = design_vals['free_stream']

# read inflow parameters
inlet_dir = main_dir + '/inlet'
os.chdir(inlet_dir)
f = open('inlet_inflow.json')
inflow = json.load(f)
f.close()

# generate Busemann field
buse_vals = {
    'M1': inflow['mach_no'],
    'p3_p1': inlet_vals['exit_press'] / inflow['press'],
    'r0': 1.0,
    'gamma': free_stream['rat_spec_heats'],
    'dtheta': 0.05*pi/180,
    'M2_guess': 5.664,
    'beta2_guess': 0.2340,
    'max_steps': 10000,
    'print_freq': 500,
    'verbosity': 1
}
field = busemann_M1_p3p1(buse_vals)

# truncate contour
field.Streamline = field.Streamline.truncate(
    trunc_angle=inlet_vals['trunc_angle'])

# import capture shape
with open('cap_shape.csv', 'r') as csvfile:
    file = csv.reader(csvfile, delimiter=' ')
    next(file)
    cap_shape = []
    for row in file:
        cap_shape.append([float(row[0]), float(row[1])])
cap_shape = np.array(cap_shape)

# translate capture shape and save as csv
min_y = np.min(cap_shape[:,1])
range_y = np.ptp(cap_shape[:,1])
cap_shape[:,1] += abs(min_y) + 0.05*range_y
with open('cap_shape_stream.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow(["x", "y"])
    for i in range(len(cap_shape)):
        writer.writerow([cap_shape[i][0], cap_shape[i][1]])

# find point on cap shape with max radius
cap_radii = np.zeros_like(cap_shape[:,1])
for i in range(len(cap_shape)):
    cap_radii[i] = sqrt(cap_shape[i][0]**2 + cap_shape[i][1]**2)

# scale and translate
scaler = 1.05 * np.max(cap_radii) / field.Streamline.ys[0]
field.Streamline = field.Streamline.scale(scaler, scaler, scaler)
z_shift = abs(field.Streamline.zs[0])
field.Streamline = field.Streamline.translate(z_shift=z_shift)

# save plot and csv of contour
field.plot(file_name='trunc_buse', show_mach_wave=False, show_exit_shock=False)
field.Streamline.save_to_csv('trunc_buse')

# generate gas input file
diffuser_dir = main_dir + '/diffuser'
if not os.path.exists(diffuser_dir):
    os.mkdir(diffuser_dir)
os.chdir(diffuser_dir )
with open('ideal-air.inp', 'w') as f:
    f.write('model = "IdealGas"\n' + "species = {'air'}")