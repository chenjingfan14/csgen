"""
Generate conical forebody geometry.

Author: Reece Otto 01/05/2022
"""
import os
import json
from csgen.conical_flow import (conical_M0_thetac, 
    eval_flow_data, 
    flow_data_to_vtk, 
    avg_flow_data)
from csgen.grid import StructuredGrid
from csgen.grid import CircularArc, CoonsPatch
from nurbskit.path import Bezier
from math import pi, tan, cos, sin
import numpy as np
import csv

#------------------------------------------------------------------------------#
#                         Generate Conical Flow Field                          #
#------------------------------------------------------------------------------#
# print title
title_width = 38
print('-'*title_width)
print(f"{'Generating Forebody':^{title_width}}")
print('-'*title_width)

# import forebody design values
f = open('design_vals.json')
design_vals = json.load(f)
f.close()
free_stream = design_vals['free_stream']
forebody_vals = design_vals['forebody']
inlet_vals = design_vals['inlet']

# navigate to forebody directory
config_dir = os.getcwd()
main_dir = os.path.dirname(config_dir)
forebody_dir = main_dir + '/forebody'
if not os.path.exists(forebody_dir):
    os.mkdir(forebody_dir)
os.chdir(forebody_dir)

# define conical field params
field_vals = {
    'mach_no': free_stream['mach_no'],
    'cone_angle': forebody_vals['cone_angle'],
    'rat_spec_heats': free_stream['rat_spec_heats'],
    'beta_guess': 7.0*pi/180,
    'tol': 1.0E-6,
    'theta_step': 0.01*pi/180,
    'interp_sing': True,
    'max_steps': 100000,
    'print_freq': 40,
    'verbosity': 1
}

# generate flow field
field = conical_M0_thetac(field_vals)

# generate cone and shock surfaces
n_r = 200
n_phi = 2*4*24 + 1
fb_len = forebody_vals['length']
cone_grid = field.cone_surface(fb_len, n_r=n_r, n_phi=n_phi)
shock_grid = field.shock_surface(fb_len, n_r=n_r, n_phi=n_phi)
StructuredGrid(cone_grid).export_to_vtk_xml(file_name='cone_surface')
StructuredGrid(shock_grid).export_to_vtk_xml(file_name='shock_surface')

#------------------------------------------------------------------------------#
#                             Evaluate Exit Flow                               #
#------------------------------------------------------------------------------#
# navigate to inlet directory
inlet_dir = main_dir + '/inlet'
if not os.path.exists(inlet_dir):
    os.mkdir(inlet_dir)
os.chdir(inlet_dir)

# create capture shape for one engine module
r_cone = fb_len*tan(field.thetac)
r_shock = fb_len*tan(field.beta)
mod_angle = inlet_vals['smile_angle']/inlet_vals['no_modules']
right_angle = 3*pi/2 + mod_angle/2
left_angle = 3*pi/2 - mod_angle/2

lower_right = [r_shock*cos(right_angle), r_shock*sin(right_angle), fb_len]
upper_right = [r_cone*cos(right_angle), r_cone*sin(right_angle), fb_len]
lower_left = [r_shock*cos(left_angle), r_shock*sin(left_angle), fb_len]
upper_left = [r_cone*cos(left_angle), r_cone*sin(left_angle), fb_len]

east = Bezier(P=[lower_right, upper_right])
west = Bezier(P=[lower_left, upper_left])
south = CircularArc(lower_left, lower_right)
north = CircularArc(upper_left, upper_right)
cap_surf = CoonsPatch(north, south, east, west)

# evaluate flow field over capture shape for engine module
n_i = 25
n_j = n_i
cap_mesh = cap_surf.grid_eval(n_s=n_i, n_t=n_j)
exit_data = eval_flow_data(field, cap_mesh, free_stream)
flow_data_to_vtk(exit_data, file_name='inlet_flow')
inlet_flow = avg_flow_data(exit_data)

# replace average xyz coords and theta with attachment coords
inlet_flow['x'] = 0.0
inlet_flow['y'] = exit_data['y'][-1][n_j//2]
inlet_flow['z'] = fb_len
inlet_flow['theta'] = exit_data['theta'][-1][n_j//2]

# export inlet inflow conditions
with open('inlet_flow.json', 'w') as f:
    json.dump(inlet_flow, f, ensure_ascii=False, indent=2)

# export capture shape
west_top_bndry = cap_mesh[n_i//2:,0]
north_bndry = cap_mesh[-1,1:]
east_bndry = cap_mesh[:-1,-1][::-1]
south_bndry = cap_mesh[0,1:-1][::-1]
west_bot_bndry = cap_mesh[:n_i//2+1,0]
cap_shape = np.concatenate((west_top_bndry, north_bndry, east_bndry, 
                            south_bndry, west_bot_bndry))
with open('cap_shape.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow(["x", "y", "z"])
    for i in range(len(cap_shape)):
        writer.writerow([cap_shape[i][0], cap_shape[i][1], fb_len])