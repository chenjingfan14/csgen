"""
Script to drive the optimization of the truncated Busemann contour.

Author: Reece Otto 14/02/2022
"""
from pyffd.ffd_utils import auto_hull_2D
from pyffd.point_inversion import point_inv_surf
from pyffd.visualisation import surf_plot_2D, path_plot_2D
from pyffd.surface import NURBSSurface
from pyffd.spline_fitting import fit_bspline_path
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from subprocess import Popen, DEVNULL
import os
from math import atan, pi

# TODO extract these vals automatically
p1 = 2612.39371900317
M1 = 7.950350403130542
T1 = 328.5921740025843

#------------------------------------------------------------------------------#
#                         Geometry parameterization                            #
#------------------------------------------------------------------------------#
print('Parameterizing baseline contour.\n')

with open('solutions/trunc_buse_0/trunc_contour_0.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    coords = list(reader)

n_points = len(coords)
contour = np.nan * np.ones((n_points, 2))
for i in range(n_points):
    contour[i][0] = float(coords[i][0])
    contour[i][1] = float(coords[i][1])

"""
# parameterize using a B-Spline path and plot
# take subset of point data as initial guess for control point positions
from math import ceil
from pyffd.utils import auto_knot_vector
from pyffd.path import BSpline
p=3
N_P = 10
Q = contour
delta_ind = (len(Q) - 1) / (N_P - 1)
P_guess = np.nan * np.ones((N_P, len(Q[0])))
for i in range(N_P):
    ind = int(ceil(i * delta_ind))
    P_guess[i] = Q[ind]
    
# create initial B-Spline path
U = auto_knot_vector(N_P, p)
spline = BSpline(P=P_guess, p=p, U=U)
plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
ax = path_plot_2D(spline, show_knots=False, path_label='Fitted B-Spline')
ax.scatter(contour[:,0], contour[:,1], label='Point Data')
plt.axis('equal')
plt.grid()
plt.legend()
plt.savefig('solutions/trunc_buse_0/contour_0.svg', bbox_inches='tight')
plt.show()
"""

# plot initial contour and FFD hull
N_Px = 4
N_Py = 4
hull = auto_hull_2D(contour, N_Px, N_Py).cast_to_nurbs_surface()
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 16
        })
ax = surf_plot_2D(hull)
ax.plot(contour[:,0], contour[:,1], 'k-', label='Contour')
ax.set_aspect('equal', adjustable="datalim")
plt.legend()
fig.savefig('solutions/trunc_buse_0/contour_0.svg', bbox_inches='tight')

# run point inversion routine
params = np.nan * np.ones(contour.shape)
for i in range(len(contour)):
    params[i] = point_inv_surf(hull, contour[i], tol=1E-6)
print('Geometry parameterization complete. \n')

#------------------------------------------------------------------------------#
#                            Objective function                                #
#------------------------------------------------------------------------------#
def perturb_hull(design_vars, hull):
    """
    # control points and weights
    no_weights = len(hull.G) * len(hull.G[0])
    G_new = np.reshape(design_vars[-no_weights:], hull.G.shape)
    P_new = np.reshape(design_vars[:-no_weights], hull.P.shape)
    hull_new = NURBSSurface(P=P_new, G=G_new, p=hull.p, q=hull.q, U=hull.U, 
        V=hull.V)

    # just control points
    P_new = np.reshape(design_vars, hull.P.shape)
    hull_new = NURBSSurface(P=P_new, G=hull.G, p=hull.p, q=hull.q, U=hull.U, 
        V=hull.V)
    """
    # geometric constraints
    global design_xs
    global no_xs
    global design_ys
    global no_ys
    global design_Gs
    global no_Gs
    global N_Py
    global throat_y

    P_new = hull.P
    var_xs = design_vars[:no_xs].reshape(hull.P[1:-1][:,:,0].shape)
    var_ys_flat = np.insert(design_vars[no_xs:no_xs+no_ys], -N_Py+1, throat_y)
    var_ys = var_ys_flat.reshape(hull.P[:,:,1].shape)
    P_new[1:-1][:,:,0] = var_xs
    P_new[:,:,1] = var_ys
    """
    return NURBSSurface(P=P_new, G=hull.G, p=hull.p, q=hull.q, U=hull.U, 
        V=hull.V)
    """
    # weights
    var_Gs = design_vars[no_xs+no_ys:no_xs+no_ys+no_Gs].reshape(hull.G.shape)
    G_new = var_Gs
    return NURBSSurface(P=P_new, G=G_new, p=hull.p, q=hull.q, U=hull.U, 
        V=hull.V)

def perturb_contour(hull, params):
    contour_new = np.nan * np.ones(params.shape)
    for i in range(len(params)):
        contour_new[i] = hull(params[i][0], params[i][1])
    return contour_new

def perturb_spline(design_vars, spline):
    P_new = np.reshape(design_vars, spline.P.shape)
    spline_new = BSpline(P=P_new, p=p, U=U)
    return spline_new

def sim_gen(sol_no, contour):
    # generate directory
    path = os.getcwd()
    base_dir = path + '/solutions/trunc_buse_' + str(sol_no) + '/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # create gas model input file
    gas_file = base_dir + 'ideal-air.inp'
    f = open(gas_file, 'w')
    f.write('model = "IdealGas" \nspecies = {\'air\'}')
    f.close()
    os.chmod(gas_file, 0o777)

    # create puffin script
    puffin_txt = f"""config.axisymmetric = True
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
with open('trunc_contour_{sol_no}.csv', 'r') as df:
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
    puffin_file = base_dir + 'trunc_buse_' + str(sol_no) + '.py'
    f = open(puffin_file, 'w')
    f.write(puffin_txt)
    f.close()
    os.chmod(puffin_file, 0o777)

    # save truncated contour as a CSV file
    contour_file = base_dir + 'trunc_contour_' + str(sol_no) + '.csv'
    with open(contour_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for i in range(len(contour)):
            writer.writerow([contour[i][0], contour[i][1]])
    os.chmod(contour_file, 0o777)

    # save bash bash script to run simulation
    bash_txt = f"""#!/bin/bash
prep-gas ideal-air.inp ideal-air-gas-model.lua
puffin-prep --job=trunc_buse_{sol_no}
puffin --job=trunc_buse_{sol_no}
puffin-post --job=trunc_buse_{sol_no} --output=vtk
puffin-post --job=trunc_buse_{sol_no} --output=stream --cell-index=$ --stream-index=0
    """ 
    bash_file = base_dir + 'run-trunc_buse_' + str(sol_no) + '.sh'
    f = open(bash_file, 'w')
    f.write(bash_txt)
    f.close()
    os.chmod(bash_file, 0o777)

    return base_dir

def obj_fcn(p_tgt, contour, sol_no):
    # calculate x value at throat
    x_throat = contour[-1][0]

    # extract data from current flow solution
    sol_dir = 'solutions/trunc_buse_' + str(sol_no) + '/trunc_buse_' + \
               str(sol_no) + '/'
    with open(sol_dir + 'flow-0.data', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        data = list(reader)

    # sum square of static pressure residual
    J = 0
    for i in range(1, len(data)):
        if float(data[i][0]) >= x_throat:
            J += (float(data[i][6]) - p_tgt)**2
    
    # normalise objective function against baseline solution
    global J0
    return J/J0

def plot_geom(hull_new, contour_new, base_dir, sol_no):
    fig = plt.figure(figsize=(16, 9))
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 16
        })
    ax = surf_plot_2D(hull_new)
    ax.plot(contour_new[:,0], contour_new[:,1], 'k-', label='Contour')
    ax.set_aspect('equal', adjustable="datalim")
    plt.legend()
    geom_file = base_dir + '/contour_' + str(sol_no) + '.svg'
    fig.savefig(geom_file, bbox_inches='tight')
    plt.close(fig)

def plot_spline(spline_new, contour_new, base_dir, sol_no):
    fig = plt.figure(figsize=(16, 9))
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 16
        })
    ax = path_plot_2D(spline_new, show_knots=False)
    #ax.plot(contour_new[:,0], contour_new[:,1], 'k-', label='Contour')
    ax.set_aspect('equal', adjustable="datalim")
    plt.legend()
    geom_file = base_dir + '/contour_' + str(sol_no) + '.svg'
    fig.savefig(geom_file, bbox_inches='tight')
    plt.close(fig)

def driver(design_vars, p_tgt, contour, hull, params):
    # generate new hull based perturbed design variables
    hull_new = perturb_hull(design_vars, hull)

    # generate new contour based on perturbed hull
    contour_new = perturb_contour(hull_new, params)

    # generate simulation files
    global sol_no
    base_dir = sim_gen(sol_no, contour_new)

    # run flow solver
    bash_command = f'./run-trunc_buse_{sol_no}.sh'
    p = Popen(bash_command, cwd=base_dir , stdout=DEVNULL)
    #p = Popen(bash_command, cwd=base_dir)
    p_status = p.wait()

    # evaluate objective function
    J = float(obj_fcn(p_tgt, contour_new, sol_no))

    # print current objective function eval number and value
    print(f'Eval={sol_no} J={J:.4} D={design_vars}')

    # create SVG of geometry
    plot_geom(hull_new, contour_new, base_dir, sol_no)

    # add 1 to the solution number for next iteration
    sol_no += 1

    return J

def driver_spline(design_vars, p_tgt, contour, spline):
    # generate new spline based perturbed design variables
    spline_new = perturb_spline(design_vars, spline)

    # generate new contour based on perturbed hull
    contour_new = spline_new.list_eval(n_points=len(contour))

    # generate simulation files
    global sol_no
    base_dir = sim_gen(sol_no, contour_new)

    # run flow solver
    bash_command = f'./run-trunc_buse_{sol_no}.sh'
    p = Popen(bash_command, cwd=base_dir, stdout=DEVNULL)
    #p = Popen(bash_command, cwd=base_dir)
    p_status = p.wait()

    # evaluate objective function
    J = float(obj_fcn(p_tgt, contour_new, sol_no))

    # print current objective function eval number and value
    print(f'Eval={sol_no} J={J:.4} D={design_vars}')

    # create SVG of geometry
    plot_spline(spline_new, contour_new, base_dir, sol_no)

    # add 1 to the solution number for next iteration
    sol_no += 1

    return J

"""
construct vector of design variables
geometric constraints:
    fix x position of first column of control points
    fix x position of last column of control points
    fix y position of bottom right control point

note: throat point should remain fixed to ensure objective function is summed
over the exact same cells on each iteration (otherwise design space is 
changing every iteration -> potential convergence issues)
"""
design_xs = hull.P[1:-1][:,:,0].flatten()
no_xs = len(design_xs)
design_ys = hull.P[:,:,1].flatten()
throat_y = design_ys[-N_Py]
design_ys = np.delete(design_ys, -N_Py)
no_ys = len(design_ys)

design_Gs = hull.G.flatten()
no_Gs = len(design_Gs)
design_vars = np.concatenate((design_xs, design_ys, design_Gs))

#design_vars = np.concatenate((design_xs, design_ys))
# calculate objective function for iteration 0
sol_no = 0
p_tgt = 50E3
J0 = 1
J0 = obj_fcn(p_tgt, contour, sol_no)

# set bounds for design variables
lb = np.zeros(len(design_vars))
ub = np.zeros(len(design_vars))
for i in range(len(lb)):
    lb[i] = design_vars[i] - 0.2
    ub[i] = design_vars[i] + 0.2
bounds = Bounds(lb, ub)


# constraints
def init_angle(design_vars):
    print(abs(atan((design_vars[no_xs+7]-design_vars[no_xs+3]) / \
        design_vars[3]) * 180/pi))
    return abs(atan((design_vars[no_xs+7]-design_vars[no_xs+3]) / \
        design_vars[3]) * 180/pi)
lb_con = np.array([0])
ub_con = np.array([10*pi/180])
cons = NonlinearConstraint(init_angle, lb_con, ub_con)

# run optimizer
sol = minimize(driver, design_vars, bounds=bounds, 
    constraints={'type':'ineq', 'fun':init_angle}, method='SLSQP', 
    args=(p_tgt, contour, hull, params), options={'disp':True, 'eps':1E-5})

"""
design_vars = spline.P.flatten()
sol_no = 0
p_tgt = 50E3
sol = minimize(driver_spline, design_vars, method='Nelder-Mead', 
    args=(p_tgt, contour, spline), options={'disp':True})
"""
