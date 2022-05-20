"""
Adjusting capture shape for forebody-inlet integration.

Author: Reece Otto 14/05/2022
"""
import os 
from csgen.stream_utils import shock_surface
from csgen.grid import StructuredGrid, CircularArc
from nurbskit.spline_fitting import global_curve_interp, global_surf_interp
from nurbskit.path import Bezier, BSpline
from nurbskit.surface import BSplineSurface, NURBSSurface
from nurbskit.utils import auto_knot_vector
from nurbskit.geom_utils import rotate_x, translate
from nurbskit.cad_output import nurbs_surf_to_iges
import csv
from scipy import optimize
import numpy as np
import json
from math import pi, sin, cos, tan, sqrt

#------------------------------------------------------------------------------#
#                          Adjusting Capture Shape                             #
#------------------------------------------------------------------------------#
# cosntruct inlet shock surface
config_dir = os.getcwd()
main_dir = os.path.dirname(config_dir)
diffuser_dir = main_dir + '/diffuser'
inlet_dir = main_dir + '/inlet'
os.chdir(inlet_dir)
diffuser_vals = {
    'puffin_dir': diffuser_dir,
    'job_name': 'diffuser',
    'n_r': 100,
    'n_phi': 100,
    'p_rat_shock': 2.0,
    'dt': 1.0E-5,
    'max_step': 100000
}
print('Constructing inlet shock surface from puffin solution...')
inlet_shock = shock_surface(diffuser_vals)
StructuredGrid(inlet_shock).export_to_vtk_xml(file_name='inlet_shock_raw')
poly_degree = 3
U_shock, V_shock, P_shock = global_surf_interp(inlet_shock, poly_degree, 
                                               poly_degree)
shock_spline = BSplineSurface(p=poly_degree, q=poly_degree, U=U_shock, 
                              V=V_shock, P=P_shock)
print('Done.')

# find inlet attach point by projecting capture shape onto inlet shock surface
cap_shape = []
with open('cap_shape_stream.csv', 'r') as csvfile:
    file = csv.reader(csvfile, delimiter=' ')
    next(file)
    cap_shape = []
    for row in file:
        cap_shape.append([float(row[0]), float(row[1])])
attach_point = np.array(cap_shape[len(cap_shape)//4])

def dist_attach(params, shock_spline, attach_point):
    u = params[0]
    v = params[1]
    return np.linalg.norm(attach_point - shock_spline(u, v)[:2])

bounds = ((0, 1), (0, 1))
sol = optimize.minimize(dist_attach, [0.5, 0.5], 
    args=(shock_spline, attach_point), method='SLSQP', bounds=bounds)
if not sol.success or sol.fun > 1.0E-5:
    raise AssertionError('Optimizer failed to locate inlet attachment point.')
inlet_attach = shock_spline(sol.x[0], sol.x[1])
inlet_attach[0] = 0.0

# tranform inlet shock to correct orientation
os.chdir(diffuser_dir)
f = open('inlet_inflow.json')
forebody_attach = json.load(f)
f.close()
os.chdir(inlet_dir)

y_shift = -inlet_attach[1] + forebody_attach['y']
z_shift = -inlet_attach[2] + forebody_attach['z']
P_shock_trans = translate(P_shock, y_shift=y_shift, z_shift=z_shift)
P_shock_trans = rotate_x(P_shock_trans, angle=forebody_attach['theta'], 
    x_origin=1.0, y_origin=forebody_attach['y'], z_origin=forebody_attach['z'])
shock_spline_trans = BSplineSurface(p=poly_degree, q=poly_degree, U=U_shock, 
                                    V=V_shock, P=P_shock_trans)
inlet_shock_trans = shock_spline_trans.list_eval(N_u=len(inlet_shock), 
                                                 N_v=len(inlet_shock[0]))
StructuredGrid(inlet_shock_trans).export_to_vtk_xml(file_name='inlet_shock')

# create NURBS patch for quarter-section of cone
back_z = forebody_attach['z']
back_rad = back_z * tan(forebody_attach['theta'])
mid_z = back_z / 2
mid_rad = back_rad / 2

P_cone = [[[-mid_rad, 0.0, mid_z], [-mid_rad, -mid_rad, mid_z], 
           [0.0, -mid_rad, mid_z]],
              
          [[-back_rad, 0.0, back_z], [-back_rad, -back_rad, back_z], 
           [0.0, -back_rad, back_z]]]
G_cone = [[1, 1/sqrt(2), 1],
          [1, 1/sqrt(2), 1]]
p_cone = 1
q_cone = 2
U_cone = auto_knot_vector(len(P_cone), p_cone)
V_cone = auto_knot_vector(len(P_cone[0]), q_cone)
cone_sec = NURBSSurface(P=P_cone, G=G_cone, p=p_cone, q=q_cone, 
                        U=U_cone, V=V_cone)
cone_sec_grid = cone_sec.list_eval()
StructuredGrid(cone_sec_grid).export_to_vtk_xml(file_name='cone_sec')

# calculate intersection contour between cone_sec and shock_spline
def dist_intersect(params, surf_1, surf_2, bndry, u1):
    if bndry == 'left':
        u1 = 0
        v1 = params[0]
    elif bndry == 'right':
        u1 = params[0]
        v1 = 1
    else:
        u1 = u1
        v1 = params[0]
    u2 = params[1]
    v2 = params[2]
    return np.linalg.norm(surf_1(u1, v1) - surf_2(u2, v2))

# solve for parameters at contour midpoint
bounds = ((0, 1), (0, 1), (0, 1))
sol_mid = optimize.minimize(dist_intersect, [0.5, 1.0, 1.0], 
    args=(shock_spline_trans, cone_sec, 'left', None), method='SLSQP', 
    bounds=bounds)
if not sol_mid.success or sol_mid.fun > 1.0E-5:
    raise AssertionError('Optimizer failed to locate mid-point of intersection '
        'contour.')

# solve for parameters at edge of contour
sol_end = optimize.minimize(dist_intersect, [0.5, 0.5, 0.5], 
    args=(shock_spline_trans, cone_sec, 'right', None), method='SLSQP', 
    bounds=bounds)
if not sol_end.success or sol_end.fun > 1.0E-5:
    raise AssertionError('Optimizer failed to locate edge of intersection '
        'contour.')

# solve for parameters along contour
n_points = 20
u1s = np.linspace(0, sol_end.x[0], n_points)
v1_guesses = np.linspace(sol_mid.x[0], 1, n_points)
valid_points = []

print('Calculating intersection contour...')
for i in range(n_points):
    sol_end = optimize.minimize(dist_intersect, [v1_guesses[i], 0.5, 0.5], 
    args=(shock_spline_trans, cone_sec, 'mid', u1s[i]), method='SLSQP', 
    bounds=bounds)

    if sol_end.success and sol_end.fun <= 1.0E-5:
        coords = cone_sec(sol_end.x[1], sol_end.x[2])
        valid_points.append(coords)
        print(f'point {i}: dist={sol_end.fun:.4e}, coords=[{coords[0]:.3f}, '
              f'{coords[1]:.3f}, {coords[2]:.3f}]')

# fit B-Spline to intersection curve
valid_points[0] = [0.0, forebody_attach['y'], forebody_attach['z']]
contour_left = np.array(valid_points[::-1])
p_int = 3
U_int, P_int = global_curve_interp(contour_left, p_int)
int_spline = BSpline(P=P_int, U=U_int, p=p_int)

# find where capture shape side-wall connects to intersection contour
os.chdir(config_dir)
f = open('design_vals.json')
design_vals = json.load(f)
f.close()
os.chdir(inlet_dir)
inlet_vals = design_vals['inlet']
forebody_vals = design_vals['forebody']
fb_len = forebody_vals['length']
r_cone = fb_len*tan(forebody_vals['cone_angle'])
mod_angle = inlet_vals['smile_angle']/inlet_vals['no_modules']
left_angle = (3*pi - mod_angle)/2

def left_sidewall(x):
    return x*tan(left_angle)

def dist_line(params, curve):
    u = params[0]
    x = params[1]
    line_point = np.array([x, left_sidewall(x)])
    return np.linalg.norm(curve(u)[:2] - line_point)

bounds = ((0, 1), (r_cone*cos(left_angle), 0.0))
sol_corn = optimize.minimize(dist_line, [0.5, r_cone*cos(left_angle)/2], 
    args=(int_spline), method='SLSQP', bounds=bounds)
if not sol_corn.success or sol_corn.fun > 1.0E-5:
    raise AssertionError('Optimizer failed to project the sidewall of capture '
        'to intersection contour.')

# adjust top boundary of capture shape
proj_point = int_spline(sol_corn.x[0])
valid_points = [point for point in valid_points if point[0] > proj_point[0]]
valid_points.append(proj_point)
cap_top_left = np.array(valid_points[::-1])
cap_top_right = np.array(valid_points[1:])
for i in range(len(cap_top_right)):
    cap_top_right[i][0] *= -1
cap_top = np.concatenate((cap_top_left, cap_top_right))
p_north = 3
U_north, P_north = global_curve_interp(cap_top, p_north)
north = BSpline(P=P_north, U=U_north, p=p_north)

# adjust remaining boundaries
right_angle = (3*pi + mod_angle)/2
cap_shape = []
with open('cap_shape.csv', 'r') as csvfile:
    file = csv.reader(csvfile, delimiter=' ')
    next(file)
    cap_shape = []
    for row in file:
        cap_shape.append([float(row[0]), float(row[1])])
r_shock = abs(np.min(np.array(cap_shape)[:,1]))
lower_right = [r_shock*cos(right_angle), r_shock*sin(right_angle), fb_len]
upper_right = proj_point.copy()
upper_right[0] *= -1
lower_left = [r_shock*cos(left_angle), r_shock*sin(left_angle), fb_len]
upper_left = proj_point
east = Bezier(P=[lower_right, upper_right])
west = Bezier(P=[lower_left, upper_left])
south = CircularArc(lower_left, lower_right)

# discretize capture shape
n_i = 21
n_j = 21
west_upper_disc = west.list_eval(u_i=0.5, n_points=n_j//2)
north_disc = north.list_eval(n_points=n_i)
east_disc = east.list_eval(n_points=n_j)[::-1]
south_disc = south.list_eval(n_points=n_i)[::-1]
west_lower_disc = west.list_eval(u_f=0.5, n_points=n_j//2)
cap_shape_adj = np.concatenate((west_upper_disc[:-1], north_disc[:-1], 
    east_disc[:-1], south_disc[:-1], west_lower_disc))

# export north boundary as csv
with open('intersect_contour.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow(["x", "y", "z"])
    for i in range(len(north_disc)):
        writer.writerow([north_disc[i][0], north_disc[i][1], north_disc[i][2]])

# rotate capture shape to reference frame of diffuser (rotate about attach pt)
rot_angle = -forebody_attach['theta']
R_x = np.array([[1.0, 0.0, 0.0],
                [0.0, cos(rot_angle), -sin(rot_angle)],
                [0.0, sin(rot_angle), cos(rot_angle)]])
for i in range(len(cap_shape_adj)):
    cap_shape_adj[i][1] -= forebody_attach['y']
    cap_shape_adj[i][2] -= forebody_attach['z']
    
    cap_shape_adj[i] = np.matmul(R_x, cap_shape_adj[i])

    cap_shape_adj[i][1] += forebody_attach['y']
    cap_shape_adj[i][2] += forebody_attach['z']

# save to csv
with open('cap_shape_adj.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow(["x", "y"])
    for i in range(len(cap_shape_adj)):
        writer.writerow([cap_shape_adj[i][0], cap_shape_adj[i][1]])

# trim conical forebody geometry
P_fb = np.zeros((2, len(P_north), 3))
P_fb[1] = P_north.copy()
p_fb = 1
q_fb = p_north
U_fb = auto_knot_vector(len(P_fb), p_fb)
V_fb = U_north.copy()
fb_trim = BSplineSurface(p=p_fb, q=q_fb, U=U_fb, V=V_fb, P=P_fb)
fb_trim = fb_trim.cast_to_nurbs_surface()
nurbs_surf_to_iges(fb_trim, file_name='forebody_trimmed')
fb_trim_disc = fb_trim.list_eval(N_u=50, N_v=n_i)
StructuredGrid(fb_trim_disc).export_to_vtk_xml(file_name='forebody_trimmed')