"""
Optimization of a nose cone with a volume constraint.

Author: Reece Otto 30/05/2022
"""
from math import sin, atan, sqrt, acos
from nurbskit.surface import NURBSSurface
from nurbskit.visualisation import surf_plot_3D
from nurbskit.utils import auto_knot_vector
from nurbskit.file_io import surf_to_vtk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pyoptsparse import SLSQP, Optimization, History
import copy

# construct initial nose cone shape with NURBS
a = 1; b = 1; L = 5; n_i = 5
P_ellipse = np.array([[-a, 0.0, L], [-a, b, L], [0.0, b, L], [a, b, L], 
    [a, 0.0, L], [a, -b, L], [0.0, -b, L], [-a, -b, L], [-a, 0.0, L]])
G_ellipse = np.array([1, 1, 2, 1, 1, 1, 2, 1, 1])

P_cone = np.zeros((n_i, len(P_ellipse), 3))
G_cone = np.zeros((n_i, len(G_ellipse)))
for i in range(n_i):
    P_cone[i] = i/(n_i - 1) * P_ellipse
    if i == 1:
        for j in range(len(P_cone[0])):
            P_cone[i,j,2] = 0.0
    G_cone[i] = G_ellipse

p_cone = 4
q_cone = 2
U_cone = auto_knot_vector(len(P_cone), p_cone)
V_cone = [0, 0, 0, 1/4, 1/4, 1/2, 1/2, 3/4, 3/4, 1, 1, 1]
nose_cone = NURBSSurface(P=P_cone, G=G_cone, 
                         p=p_cone, q=q_cone, 
                         U=U_cone, V=V_cone)

# construct frustrum
r_b = 0.75
P_circle = np.array([[-r_b, 0.0, L], [-r_b, r_b, L], [0.0, r_b, L], 
    [r_b, r_b, L], [r_b, 0.0, L], [r_b, -r_b, L], [0.0, -r_b, L], 
    [-r_b, -r_b, L], [-r_b, 0.0, L]])

P_frust = np.zeros((2, len(P_circle), 3))
G_frust = np.zeros((2, len(P_circle)))
P_frust[0] = 0.5*P_circle; P_frust[1] = P_circle
G_frust[0] = G_ellipse; G_frust[1] = G_ellipse
p_frust = 1; q_frust = 2
U_frust = auto_knot_vector(len(P_frust), p_frust)
frustrum = NURBSSurface(P=P_frust, G=G_frust, 
                        p=p_frust, q=q_frust, 
                        U=U_frust, V=V_cone)

# export surfaces as VTK
N_u = 100; N_v = 101
surf_to_vtk(nose_cone, file_name='init_nose', N_u=N_u, N_v=N_v)
surf_to_vtk(frustrum, file_name='frustrum', N_u=N_u, N_v=N_v)

def perturb_cone(design_vars):
    """
    Deforms the shape of a nose cone for a new set of design variables.

    Parameters:
        design_vars (dict): design variables

    Returns:
        deformed_hull (NURBSSurface): deformed nose cone
    """
    deformed_hull = copy.deepcopy(nose_cone)
    for key in design_vars:
        split_name = key.split('_')
        i = int(split_name[1]); j = int(split_name[2])
        if split_name[0] == 'x':
            deformed_hull.P[i][j][0] = design_vars[key]
        if split_name[0] == 'y':
            deformed_hull.P[i][j][1] = design_vars[key]
    
    # ensure surface remains closed
    deformed_hull.P[:,-1] = deformed_hull.P[:,0]

    return deformed_hull

def drag_coeff_grid(grid):
    """
    Calculates the total drag coefficient on a structured grid using Newtonian
    flow.

    Parameters:
        grid (np.ndarray): structured grid

    Returns:
        (float): total drag coefficient
    """
    n_i = len(grid); n_j = len(grid[0])
    c_d_sum = 0.0
    n_lines = 0 
    for i in range(n_i-1):
        for j in range(n_j):
            del_x = grid[i+1][j][0] - grid[i][j][0]
            del_y = grid[i+1][j][1] - grid[i][j][1]
            del_z = grid[i+1][j][2] - grid[i][j][2]
            del_r = sqrt(del_x**2 + del_y**2 + del_z**2)
            theta = acos(del_z/del_r)
            c_d_sum += 2*(sin(theta))**3
            n_lines += 1
    return c_d_sum / n_lines

def vol_con(grid):
    """
    Calculates penalty when a given grid enters the frustrum volume.

    Parameters:
        grid (np.ndarray): structured grid

    Returns:
        penalty (float): penalty for entering frustrum
    """
    n_i = len(grid); n_j = len(grid[0])
    penalty = 0.0
    for i in range(n_i):
        for j in range(n_j):
            z = grid[i][j][2]
            if z >= L/2:
                r_surf = sqrt(grid[i][j][0]**2 + grid[i][j][1]**2)
                r_min = r_b/L * (z-L/2) + r_b/2
                if r_surf < r_min:
                    penalty += (r_min - r_surf)**2
    return penalty

def driver(design_vars):
    """
    Function to drive optimizer.

    Parameters:
        design_vars (dict): design variables

    Returns:
        funcs (dict): objective and constraint functions
        fail (bool): flag to determine if optimization has failed
    """
    # perturb cone
    deformed_cone = perturb_cone(design_vars)

    # evaluate grid for cone
    deformed_grid = deformed_cone.list_eval(N_u=20, N_v=21)

    # evaluate drag coefficient
    funcs = {}
    funcs['drag_coeff'] = drag_coeff_grid(deformed_grid)
    funcs['volume'] = vol_con(deformed_grid)
    fail = False

    return funcs, fail

# initialise optimization problem
opt_prob = Optimization('Drag Minimization of Nose Cone', driver)

# add design variables
x_tol = 0.5; y_tol = 0.5
N_Pu = len(P_cone); N_Pv = len(P_cone[0])
for i in range(1, N_Pu):
    for j in range(N_Pv-1):
        
        x_val = nose_cone.P[i][j][0]
        lower_x_val = x_val - x_tol
        upper_x_val = x_val + x_tol
        opt_prob.addVar(f'x_{i}_{j}', varType='c', value=x_val, 
                        lower=lower_x_val, upper=upper_x_val)
        
        y_val = nose_cone.P[i][j][1]
        lower_y_val = y_val - y_tol
        upper_y_val = y_val + y_tol
        opt_prob.addVar(f'y_{i}_{j}', varType='c', value=y_val, 
                        lower=lower_y_val, upper=upper_y_val)

# add objective and constraints
opt_prob.addObj('drag_coeff')
opt_prob.addCon('volume', lower=None, upper=0.0)

# solve optimization problem
opt_options = {'IPRINT': 1, 'ACC':1E-06}
opt = SLSQP(options=opt_options)
sol = opt(opt_prob, sens='FD', storeHistory='hist')
print(sol)

# save grid of optimized geometry
nose_opt = perturb_cone(sol.xStar)
surf_to_vtk(nose_opt, file_name='opt_nose', N_u=N_u, N_v=N_v)