"""
Script to drive optimizer.

Author: Reece Otto 21/05/2022
"""
import os
import csv
import copy
import shutil
import subprocess
import numpy as np
from math import atan, pi, sqrt
import matplotlib.pyplot as plt
from csgen.file_io import coords_to_csv
from nurbskit.file_io import import_nurbs_surf
from nurbskit.visualisation import surf_plot_2D
from pyoptsparse import SLSQP, Optimization 

def import_shape(ffd_file, params_file):
    """
    Import parametric coordinates of contour and initial FFD hull.

    Parameters:
        ffd_file (str): file name for initial FFD hull data
        params_file (str): file name for parametric contour data

    Returns:
        ffd_hull (NURBSurface): FFD hull as a NURBSSurface object
        contour_params (np.ndarray): parametric contour data
    """
    # import parametric contour data
    contour_params = []
    with open(params_file + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        next(csv_reader)
        for row in csv_reader:
            contour_params.append([float(row[0]), float(row[1])])
    
    return import_nurbs_surf(ffd_file), np.array(contour_params)

def perturb_hull(design_vars):
    """
    Deforms FFD hull for a new set of control point positions.

    Parameters:
        design_vars (dict): dictionary of all design variables

    Returns:
        deformed_hull (NURBSSurface): deformed FFD hull
    """
    deformed_hull = copy.deepcopy(ffd_hull_init)
    for key in design_vars:
        split_name = key.split('_')
        i = int(split_name[1]); j = int(split_name[2])
        if split_name[0] == 'y':
            deformed_hull.P[i][j][1] = design_vars[key]
        
        if split_name[0] == 'G':
            deformed_hull.G[i][j] = design_vars[key]

    return deformed_hull

def perturb_contour(deformed_hull):
    """
    Deforms contour in accordance with newly deformed FFD hull.

    Parameters:
        deformed_hull (NURBSSurface): deformed FFD hull

    Returns:
        contour_coords (np.array): deformed contour coordinates
    """
    contour_coords = np.zeros_like(contour_params)
    for i in range(len(contour_params)):
        contour_coords[i] = deformed_hull(contour_params[i][0], 
            contour_params[i][1])

    return contour_coords

def plot_geom(deformed_hull, deformed_contour, obj_eval_no):
    """
    Plots deformed contour within the deformed FFD hull.

    Parameters:
        deformed_hull (NURBSSurface): deformed FFD hull
        deformed_contour (np.array): deformed contour coordinates
        obj_eval_no (int): current optimization solution number
    """
    fig = plt.figure(figsize=(16, 9))
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 16
        })
    ax = surf_plot_2D(deformed_hull, show_knots=False, mesh_label='FFD Hull')
    ax.plot(deformed_contour[:,0], deformed_contour[:,1], color='k', 
        linestyle='-', label='Contour')
    ax.set_aspect('equal', adjustable="datalim")
    plt.legend()
    fig.savefig(f'geom_{obj_eval_no}.svg', bbox_inches='tight')
    plt.close(fig)

def run_puffin_sim(deformed_contour, deformed_hull, obj_eval_no):
    """
    Creates and runs puffins simulation for deformed contour.

    Parameters:
        deformed_contour (np.ndarray): deformed contour coordinates
        obj_eval_no (int): the current number of objective function evaluations
    """
    # navigate to 'solutions' directory
    main_dir = os.getcwd()
    os.chdir(main_dir + '/solutions')

    # create new solution directory
    sol_i_dir = f'sol_{obj_eval_no}'
    if not os.path.exists(sol_i_dir):
        os.makedirs(sol_i_dir)

    # copy relevant contents of 'sol_0' to new solution directoy
    files = ['diffuser.py', 'ideal-air.inp', 'inflow.json', 'run.sh']
    for file in files:
        shutil.copyfile(main_dir +'/sol_0/' + file, sol_i_dir + '/' + file)

    # save deformed contour to solution directory
    current_dir = main_dir + '/solutions/' + sol_i_dir
    os.chdir(current_dir)
    coords_to_csv(deformed_contour, file_name=f'contour_{obj_eval_no}')
    plot_geom(deformed_hull, deformed_contour, obj_eval_no)

    # run puffin simulation
    os.chmod('run.sh', 0o777)
    comm = subprocess.Popen("./run.sh", shell=True, stdout=subprocess.DEVNULL)
    comm.wait()

    # return back to main directory
    os.chdir(main_dir)
    
def press_penalty(p_tgt, contour, sol_no):
    """
    Calculates pressure penalty function.

    Parameters:
        p_tgt (float): target exit pressure
        contour (np.ndarray): contour coordinates
        sol_no (int): current optimization solution number

    Returns:
        (float): mass flow rate-averaged pressure penalty function
    """
    # calculate x value at throat
    x_throat = contour[-1][0]

    # extract data from current flow solution
    sol_dir = f'solutions/sol_{sol_no}/diffuser/'
    with open(sol_dir + 'flow-0.data', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        data = list(reader)

    # calculate sum of pressure penalty function
    J = 0.0
    m_dot = 0.0
    throat_flag = False
    for i in range(1, len(data)):
        if float(data[i][0]) >= x_throat:
            y_i = float(data[i][1])
            if not throat_flag:
                del_y = float(data[i+1][1]) - y_i
                throat_flag = True
            rho_i = float(data[i][5])
            v_x_i = float(data[i][2])
            A_i = 2*pi*del_y*y_i
            m_dot_i = rho_i*v_x_i*A_i
            J += m_dot_i*(float(data[i][6]) - p_tgt)**2
            m_dot += m_dot_i

    # normalise objective function against reference solution
    return J/m_dot/111185596.60639724

def init_angle(hull):
    """
    Calculates the angle between the first two control points on the first row
    of an FFD hull.

    Parameters:
        hull (NURBSSurface): FFD hull

    Returns:
        (float): initial hull angle in radians
    """
    N_Pv = len(hull.P[0])
    del_x = hull.P[1][N_Pv-1][0] - hull.P[0][N_Pv-1][0]
    del_y = hull.P[1][N_Pv-1][1] - hull.P[0][N_Pv-1][1]
    return atan(del_y/del_x)

def obj_func(d_dict):
    """
    Evaluates objective functions and constraints for given design variables.

    Parameters:
        d_dict (dict): design variables

    Returns:
        funcs (dict): objective and constraint functions
        fail (bool): flag to determine if objective function or constraint
                     evaluation has failed
    """
    global obj_eval_no

    # deform FFD hull based on new design variables
    deformed_hull = perturb_hull(d_dict)

    # deform contour in accordance with new FFD hull
    deformed_contour = perturb_contour(deformed_hull)

    # run puffin simulation for newly deformed contour
    run_puffin_sim(deformed_contour, deformed_hull, obj_eval_no)

    # evalute objective function
    funcs = {}
    p_tgt = 50E3
    funcs['p_penalty'] = press_penalty(p_tgt, deformed_contour, obj_eval_no)
    funcs['init_angle'] = init_angle(deformed_hull)
    fail = False
    
    print(f"i={obj_eval_no}, J={funcs['p_penalty']}, "
        f"init_ang={funcs['init_angle']*180/pi:.5} deg")
    obj_eval_no += 1

    return funcs, fail

def main():
    # navigate to root directory
    core_dir = os.getcwd()
    main_dir = os.path.dirname(core_dir)
    os.chdir(main_dir)

    # import contour params and initial FFD hull
    ffd_file = 'ffd_hull_0'
    params_file = 'contour_params'
    global ffd_hull_init, contour_params
    ffd_hull_init, contour_params = import_shape(ffd_file, params_file)
    N_Pu = len(ffd_hull_init.P); N_Pv = len(ffd_hull_init.P[0])

    # construct optimization problem
    opt_prob = Optimization("Axisymmetric Diffuser Optimization", obj_func)
    opt_prob.addObj('p_penalty')
    
    # choosing design variables (control point positions)
    # geometric constraints:
    # - fixed bottom row of control points
    # - fixed left column of control points
    # - fixed x coords
    # - y coords can only move +/- y_tol from original position
    y_tol = 0.10
    for i in range(N_Pu):
        for j in range(N_Pv):
            if not (i==0 and j==N_Pv-1) and not (i==N_Pu-1 and j==0):
                y_val = ffd_hull_init.P[i][j][1]
                lower_y_val = y_val - y_tol
                upper_y_val = y_val + y_tol
                opt_prob.addVar(f'y_{i}_{j}', varType='c', value=y_val, 
                    lower=lower_y_val, upper=upper_y_val)

                """
                g_val = ffd_hull_init.G[i][j]
                lower_g_val = geom_tol*g_val
                upper_g_val = (1 + geom_tol)*g_val
                opt_prob.addVar(f'G_{i}_{j}', varType='c', value=g_val, 
                    lower=lower_g_val, upper=upper_g_val)
                """

    # set optimization settings
    print(opt_prob)
    opt_prob.addCon('init_angle', lower=-5.0*pi/180, upper=5.0*pi/180)
    opt_options = {"IPRINT": 1, 'ACC':1.0E-10}
    opt = SLSQP(options=opt_options)
    global obj_eval_no
    obj_eval_no = 0
    sol = opt(opt_prob, sens='FD', storeHistory='diffuser_hist')
    print(sol)

if __name__ == '__main__':
    main()