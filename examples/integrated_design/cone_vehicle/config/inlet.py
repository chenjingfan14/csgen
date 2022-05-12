"""
Generating a shape-transition inlet.

Author: Reece Otto 25/03/2022
"""
import os 
from csgen.stream_utils import shock_surface
from csgen.grid import StructuredGrid
from nurbskit.spline_fitting import global_surf_interp
from nurbskit.surface import BSplineSurface, NURBSSurface
from nurbskit.utils import auto_knot_vector
from nurbskit.geom_utils import rotate_x, translate
import csv
from scipy import optimize
import numpy as np
import json
from math import tan, sqrt

#------------------------------------------------------------------------------#
#                 Calculating Forebody-Inlet Interface Contour                 #
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

# create other half of contour by reflecting along y-axis
contour_left = np.array(valid_points[::-1])
contour_right = np.array(valid_points[1:])
for i in range(len(contour_right)):
    contour_right[i][0] *= -1
intersect_contour = np.concatenate((contour_left, contour_right))
with open('intersect_contour.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow(["x", "y", "z"])
    for i in range(len(intersect_contour)):
        writer.writerow([intersect_contour[i][0], intersect_contour[i][1], 
                         intersect_contour[i][2]])












"""
from csgen.stream_utils import inlet_stream_trace, shock_surface
from csgen.inlet_utils import inlet_blend
from nurbskit.path import Ellipse, Rectangle
from nurbskit.visualisation import path_plot_2D
from nurbskit.geom_utils import rotate_x, translate
import pyvista as pv
import csv
import numpy as np
from math import floor, pi, sqrt, tan, cos, sin
from scipy import interpolate, optimize
import os
import json
import matplotlib.pyplot as plt
"""
#------------------------------------------------------------------------------#
#                         Inlet defined by capture shape                       #
#------------------------------------------------------------------------------#
# import inlet design values and attachment info
"""
f = open('inlet_vals.json')
inlet_vals = json.load(f)
f.close()

main_dir = os.getcwd()
puffin_dir = main_dir + '/diffuser'

shock_vals = {
    'puffin_dir': puffin_dir,
    'job_name': 'diffuser',
    'n_r': inlet_vals['surf_vals']['n_z'],
    'n_phi': inlet_vals['surf_vals']['n_phi'],
    'p_rat_shock': inlet_vals['stream_settings']['p_rat_shock'],
    'dt': inlet_vals['stream_settings']['time_step'],
    'max_step': inlet_vals['stream_settings']['max_step']
}
"""
# generate capture shape
main_dir = os.getcwd()
os.chdir(main_dir + '/waverider')
with open('exit_shape.csv', 'r') as csvfile:
    file = csv.reader(csvfile, delimiter=' ')
    coords = []
    for row in file:
        coords.append([float(row[0]), float(row[1])])
os.chdir(main_dir)

cap_coords = np.array(coords)
y_min = np.amin(cap_coords[:,1])
for i in range(len(cap_coords)):
    cap_coords[i][1] += abs(y_min) + 0.05
n_phi = len(cap_coords)

working_dir = main_dir + '/inlet'
puffin_dir = main_dir + '/diffuser'
os.chdir(working_dir)

inlet_A_vals = {
    'puffin_dir': puffin_dir,       # job name for puffin simulation
    'job_name': 'diffuser',
    'shape_coords': cap_coords,     # coordinates of capture shape
    'n_phi': n_phi,                 # number of phi points
    'n_z': 100,                     # number of z points
    'n_r': 50,                      # number of r points for shock surface
    'p_rat_shock': 2.0,             # pressure ratio for shock detection
    'dt': 1.0E-5,                   # integration time step [s]
    'max_step': 100000,             # maximum number of integration steps
    'plot_shape': True,             # option to plot capture shape
    'shape_label': 'Capture Shape', # figure label for capture shape
    'file_name_shape': 'cap_shape', # file name for capture shape plot
    'save_VTK': False,              # option to save inlet surface as VTK file
    'file_name_VTK': 'inlet_A'      # file name for VTK file
}

# run streamline tracer
inlet_A_coords = inlet_stream_trace(inlet_A_vals)


#------------------------------------------------------------------------------#
#                        Inlet defined by exit shape                           #
#------------------------------------------------------------------------------#
# generate exit shape
factor = 0.1
a_cap = 2*factor
b_cap = 1*factor
h_cap = 0.0
k_cap = 0.15
exit_shape = Ellipse(a_cap, b_cap, h_cap, k_cap)
exit_coords = exit_shape.list_eval(n_points=n_phi)

inlet_B_vals = {
    'puffin_dir': puffin_dir,
    'job_name': 'diffuser',          # job name for puffin simulation
    'shape_coords': exit_coords,     # coordinates of capture shape
    'n_phi': n_phi,                  # number of phi points
    'n_z': 100,                      # number of z points
    'p_rat_shock': 2.0,              # pressure ratio for shock detection
    'dt': 1.0E-5,                    # integration time step [s]
    'max_step': 100000,              # maximum number of integration steps
    'plot_shape': True,              # option to plot exit shape
    'shape_label': 'Exit Shape',     # figure label for exit shape
    'file_name_shape': 'exit_shape', # file name for exit shape plot
    'save_VTK': False,               # option to save inlet surface as VTK file
    'file_name_VTK': 'inlet_B'       # file name for VTK file
}

# run streamline tracer
inlet_B_coords = inlet_stream_trace(inlet_B_vals)

#------------------------------------------------------------------------------#
#                               Blended inlet                                  #
#------------------------------------------------------------------------------#
inlet_C_coords = inlet_blend(inlet_A_coords, inlet_B_coords, 2.5)

f = open('attach.json')
attach_vals = json.load(f)
f.close()
angle_attach = attach_vals['attach_angle']
y_fb = attach_vals['y_attach']
z_fb = attach_vals['z_attach']
print(y_fb)

y_inlet = inlet_C_coords[floor(len(inlet_C_coords)/4),0,1]
z_inlet = inlet_C_coords[floor(len(inlet_C_coords)/4),0,2]

y_shift = -y_inlet + y_fb
z_shift = -z_inlet + z_fb


inlet_C_coords = translate(inlet_C_coords, y_shift=y_shift, z_shift=z_shift)

inlet_C_coords = rotate_x(inlet_C_coords, angle=angle_attach, x_origin=1.0,
    y_origin=y_fb, z_origin=z_fb)

inlet_A_coords = translate(inlet_A_coords, y_shift=y_shift, z_shift=z_shift)

inlet_A_coords = rotate_x(inlet_A_coords, angle=angle_attach, x_origin=1.0,
    y_origin=y_fb, z_origin=z_fb)

inlet_B_coords = translate(inlet_B_coords, y_shift=y_shift, z_shift=z_shift)

inlet_B_coords = rotate_x(inlet_B_coords, angle=angle_attach, x_origin=1.0,
    y_origin=y_fb, z_origin=z_fb)

inlet_grid_A = pv.StructuredGrid(inlet_A_coords[:,:,0], inlet_A_coords[:,:,1], 
        inlet_A_coords[:,:,2])
inlet_grid_A.save("inlet_A.vtk")

inlet_grid_B = pv.StructuredGrid(inlet_B_coords[:,:,0], inlet_B_coords[:,:,1], 
        inlet_B_coords[:,:,2])
inlet_grid_B.save("inlet_B.vtk")

inlet_grid_C = pv.StructuredGrid(inlet_C_coords[:,:,0], inlet_C_coords[:,:,1], 
        inlet_C_coords[:,:,2])
inlet_grid_C.save("inlet_C.vtk")

print('Constructing inlet shockwave surface.')
shock_surf = shock_surface(inlet_A_vals)
shock_surf = translate(shock_surf, y_shift=y_shift, z_shift=z_shift)
shock_surf = rotate_x(shock_surf, angle=angle_attach, x_origin=1.0,
    y_origin=y_fb, z_origin=z_fb)

shock_grid = pv.StructuredGrid(shock_surf[:,:,0], shock_surf[:,:,1], 
        shock_surf[:,:,2])
shock_grid.save("inlet_shock.vtk")




"""
ATTEMPTING INTERSECTION ALGORITHM HERE
ideas: 
 1 - create a curve in region of intersection and run point projection
     algorithm for each point on the curve
"""
from nurbskit.path import BSpline
from nurbskit.surface import BSplineSurface
from nurbskit.spline_fitting import global_surf_interp, global_curve_interp
from nurbskit.point_inversion import point_inv_surf
from nurbskit.cad_output import surf_to_vtk, nurbs_surf_to_iges
os.chdir(main_dir + '/waverider')
wr_trim = np.reshape(pv.read('wr_trim.vtk').points, (51, 49, 3))
os.chdir(working_dir)



p = 3
q = 3
U, V, P1 = global_surf_interp(wr_trim, p, q)
wr_patch = BSplineSurface(p=p, q=q, U=U, V=V, P=P1)

U, V, P2 = global_surf_interp(shock_surf, p, q)
shock_patch = BSplineSurface(p=p, q=q, U=U, V=V, P=P2)

def distance(params, surf_1, surf_2, bndry, v2):
    if bndry:
        u1 = params[0]
        v1 = params[1]
        u2 = 0.0
        v2 = params[2]
    else:
        u1 = params[0]
        v1 = params[1]
        u2 = params[2]
        v2 = v2

    return np.linalg.norm(surf_1(u1, v1) - surf_2(u2, v2))

params_guess = [0.5, 0.5, 0.2]
bounds = ((0, 0.99999), (0, 0.99999), (0, 0.99999))


sol = optimize.minimize(distance, params_guess, 
    args=(shock_patch, wr_patch, True, None), method='SLSQP', bounds=bounds)
if sol.fun > 1E-5 or not sol.success:
    raise Exception('Optimizer could not find boundary point of intersection'
        'curve.')
params = sol.x
print(sol)
params_bndry = [0, params[2]]
point_bndry = wr_patch(params_bndry[0], params_bndry[1])
params_mid = point_inv_surf(wr_patch, [0.0, y_fb, z_fb], tol=1E-6)
point_mid = wr_patch(params_mid[0], params_mid[1])


print(point_bndry)


n_points = 20
v2s = np.zeros(n_points-2)
dv = (params_mid[1] - params[2])/(n_points-1)
du = 1.0/(n_points-1)
for i in range(n_points-2):
    v2s[i] = params_bndry[1] + i*dv
coords = []


coords.append(point_bndry)
for i in range(n_points-2):
    u_guess = du*(1 + i)
    params_guess = [0.5, 0.5, u_guess]
    sol = optimize.minimize(distance, params_guess, 
        args=(shock_patch, wr_patch, False, v2s[i]), method='SLSQP', 
        bounds=bounds)
    if sol.success:
        point = wr_patch(sol.x[2], v2s[i])
        if point[2] <= point_mid[2] and point[2] >= point_bndry[2] and \
        point[0] >= point_bndry[0] and sol.fun <= 1E-5:
            coords.append(point)
            print(f'Intersection point found: {point}')
    else:
        print('Optimizer failed.')
coords.append(point_mid)
coords = np.array(coords)
coords_pos = coords.copy()
for i in range(len(coords_pos)):
    coords_pos[i] = coords[-(1+i)]
    coords_pos[i][0] *= -1
coords = np.concatenate((coords, coords_pos[1:]))

plt.figure(figsize=(16, 9))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 20
    })
ax = plt.axes()
ax.scatter(coords[:,0], coords[:,2], color='black')
ax.plot(coords[:,0], coords[:,2], 'k', label='Surface Intersection')
ax.set_xlabel('$x$')
ax.set_ylabel('$z$')
plt.axis('equal')
plt.grid()
plt.legend()
plt.savefig('surf_intersect.svg', bbox_inches='tight')

p = 3
U, P = global_curve_interp(coords, p)
spline = BSpline(P=P, U=U, p=p)

os.chdir(main_dir + '/waverider')
f = open('waverider_vals.json')
waverider_vals = json.load(f)
f.close()
os.chdir(working_dir)

wr_shock_angle = waverider_vals['cone_vals']['shock_angle']
n_phi = waverider_vals['surf_vals']['n_phi']
cap_coords_top = spline.list_eval(n_points=n_phi)

cap_coords_bot = np.nan * np.ones((n_phi,3))
for i in range(n_phi):
    x_ij = cap_coords_top[i][0]
    z_ij = cap_coords_top[i][2]
    cap_coords_bot[i][0] = x_ij
    cap_coords_bot[i][1] = -sqrt((tan(wr_shock_angle)*z_ij)**2 - x_ij**2)
    cap_coords_bot[i][2] = z_ij

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(cap_coords_top[:,0], cap_coords_top[:,1], cap_coords_top[:,2])
ax.scatter(cap_coords_bot[:,0], cap_coords_bot[:,1], cap_coords_bot[:,2])


cap_shape = np.concatenate((cap_coords_top, cap_coords_bot[:-1,:][::-1]))

# rotate capture shape and translate to fit in diffuser entrance
y_min = np.amin(cap_shape[:,1])
cap_shape_rot = cap_shape.copy()
R_x = np.array([[1.0, 0.0, 0.0],
                [0.0, cos(-angle_attach), -sin(-angle_attach)],
                [0.0, sin(-angle_attach), cos(-angle_attach)]])

for i in range(len(cap_shape)):
    cap_shape_rot[i][1] -= y_fb
    cap_shape_rot[i][2] -= z_fb
    cap_shape_rot[i] = np.matmul(R_x, cap_shape_rot[i])
    cap_shape_rot[i][1] += y_fb + abs(y_min) + 0.05
    cap_shape_rot[i][2] += z_fb
cap_shape_rot = cap_shape_rot[:,:-1]

# combine the two contours to make cap shape
plt.figure(figsize=(16, 9))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 20
    })
ax = plt.axes()
ax.scatter(cap_shape[:,0], cap_shape[:,1], color='black')
ax.plot(cap_shape[:,0], cap_shape[:,1], 'k', label='Capture Shape')
ax.plot(cap_shape_rot[:,0], cap_shape_rot[:,1], 'r', label='Rotated Capture Shape')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.grid()
plt.legend()
plt.savefig('cap_shape_integrated.svg', bbox_inches='tight')

# trace through diffuser
inlet_A_vals['shape_coords'] = cap_shape_rot
inlet_A_coords = inlet_stream_trace(inlet_A_vals)
inlet_grid_A = pv.StructuredGrid(inlet_A_coords[:,:,0], inlet_A_coords[:,:,1], 
        inlet_A_coords[:,:,2])




inlet_B_coords = inlet_stream_trace(inlet_B_vals)
inlet_C_coords = inlet_blend(inlet_A_coords, inlet_B_coords, 2.5)

y_inlet = inlet_C_coords[floor(len(inlet_C_coords)/4),0,1]
z_inlet = inlet_C_coords[floor(len(inlet_C_coords)/4),0,2]
y_shift = -y_inlet + y_fb
z_shift = -z_inlet + z_fb

inlet_A_coords = translate(inlet_A_coords, y_shift=y_shift, z_shift=z_shift)

inlet_A_coords = rotate_x(inlet_A_coords, angle=angle_attach, x_origin=1.0,
    y_origin=y_fb, z_origin=z_fb)

# ensure geometry is watertight
inlet_A_coords[:n_phi+1,0][:-1] = cap_coords_top
inlet_A_coords[-1,0] = inlet_A_coords[0,0]


inlet_B_coords = translate(inlet_B_coords, y_shift=y_shift, z_shift=z_shift)
inlet_B_coords = rotate_x(inlet_B_coords, angle=angle_attach, x_origin=1.0,
    y_origin=y_fb, z_origin=z_fb)
inlet_C_coords = translate(inlet_C_coords, y_shift=y_shift, z_shift=z_shift)
inlet_C_coords = rotate_x(inlet_C_coords, angle=angle_attach, x_origin=1.0,
    y_origin=y_fb, z_origin=z_fb)

# ensure geometry is watertight
inlet_C_coords[:n_phi+1,0][:-1] = cap_coords_top
inlet_C_coords[-1,0] = inlet_C_coords[0,0]

inlet_grid_A = pv.StructuredGrid(inlet_A_coords[:,:,0], inlet_A_coords[:,:,1], 
        inlet_A_coords[:,:,2])
inlet_grid_A.save("inlet_A_int.vtk")

inlet_grid_B = pv.StructuredGrid(inlet_B_coords[:,:,0], inlet_B_coords[:,:,1], 
        inlet_B_coords[:,:,2])
inlet_grid_B.save("inlet_B_int.vtk")

inlet_grid_C = pv.StructuredGrid(inlet_C_coords[:,:,0], inlet_C_coords[:,:,1], 
        inlet_C_coords[:,:,2])
inlet_grid_C.save("inlet_C_int.vtk")


class CoonsPatch():
    def __init__(self, west, north, east, south):
        self.west = west
        self.north = north
        self.east = east
        self.south = south

    def __call__(self, s, t):
        west_to_east = (1 - t)*self.west(s) + t*self.east(s)
        south_to_north = (1 - s)*self.south(t) + s*self.north(t)
        corners = self.west(0)*(1 - s)*(1 - t) + self.west(1)*s*(1 - t) + \
                  self.east(0)*(1 - s)*t + self.east(1)*s*t
        return west_to_east + south_to_north - corners

    def grid_eval(self, s_i=0, s_f=1, t_i=0, t_f=1, n_s=100, n_t=100):
        ss = np.linspace(s_i, s_f, n_s)
        ts = np.linspace(t_i, t_f, n_t)
        grid = np.nan * np.ones((n_s, n_t, 3))
        for i in range(n_s):
            for j in range(n_t):
                grid[i][j] = self(ss[i], ts[j])
        return grid


A_v = params_bndry[1]
D_v = 1-params_bndry[1]
B_v = (A_v + D_v)*1/3
C_v = (A_v + D_v)*2/3

west_coords = wr_patch.list_eval(u_i=0, u_f=0, v_i=A_v, v_f=B_v)[0]
north_coords = wr_patch.list_eval(u_i=0, u_f=0, v_i=B_v, v_f=C_v)[0]
east_coords = wr_patch.list_eval(u_i=0, u_f=0, v_i=D_v, v_f=C_v)[0]

p = 3
U, P = global_curve_interp(west_coords, p)
west= BSpline(P=P, U=U, p=p)
U, P = global_curve_interp(north_coords, p)
north = BSpline(P=P, U=U, p=p)
U, P = global_curve_interp(east_coords, p)
east = BSpline(P=P, U=U, p=p)

waverider_patch = CoonsPatch(west, north, east, spline)
n_phi = waverider_vals['surf_vals']['n_phi']
n_z = waverider_vals['surf_vals']['n_z']
waverider_coords = CoonsPatch(west, north, east, spline).grid_eval(n_s=n_phi, n_t=n_z)
waverider_grid = pv.StructuredGrid(waverider_coords[:,:,0], 
    waverider_coords[:,:,1], waverider_coords[:,:,2])
waverider_grid.save("waverider_trimmed.vtk")

with open('intersection.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for i in range(len(coords)):
        writer.writerow([coords[i,0], coords[i,1], coords[i,2]])


U, V, P1 = global_surf_interp(waverider_coords, p, q)
wr_spline = BSplineSurface(p=p, q=q, U=U, V=V, P=P1).cast_to_nurbs_surface()
nurbs_surf_to_iges(wr_spline, file_name='waverider')
wr_spline_dis = wr_spline.list_eval(N_u=11, N_v=11)
wr_spline_dis_grid = pv.StructuredGrid(wr_spline_dis[:,:,0], wr_spline_dis[:,:,1], 
        wr_spline_dis[:,:,2])
wr_spline_dis_grid.save("wr_check.vtk")

U, V1, P2 = global_surf_interp(inlet_C_coords[:floor(len(inlet_C_coords)/2)+1], p, q)



inlet_top_spline = BSplineSurface(p=p, q=q, U=V, V=V1, P=P2).cast_to_nurbs_surface()
nurbs_surf_to_iges(inlet_top_spline, file_name='inlet_top')
"""
inlet_spline_dis = inlet_top_spline.list_eval(N_u=11, N_v=11)
inlet_spline_dis_grid = pv.StructuredGrid(inlet_spline_dis[:,:,0], 
    inlet_spline_dis[:,:,1], inlet_spline_dis[:,:,2])
inlet_spline_dis_grid.save("inl_top_check.vtk")
"""

U, V1, P2 = global_surf_interp(inlet_C_coords[floor(len(inlet_C_coords)/2):], p, q)

inlet_bot_spline = BSplineSurface(p=p, q=q, U=V, V=V1, P=P2).cast_to_nurbs_surface()
nurbs_surf_to_iges(inlet_bot_spline, file_name='inlet_bot')
"""
inlet_spline_dis = inlet_bot_spline.list_eval(N_u=11, N_v=11)
inlet_spline_dis_grid = pv.StructuredGrid(inlet_spline_dis[:,:,0], 
    inlet_spline_dis[:,:,1], inlet_spline_dis[:,:,2])
inlet_spline_dis_grid.save("inl_bot_check.vtk")
"""