"""
Tools for creating Busemann flow fields.

Author: Reece Otto 03/11/2021

Station number convention:
    station 1: up-stream of entrance Mach wave
    station 2: immediately up-stream of terminating shock
    station 3: down-stream of termination shock
"""
import numpy as np
from math import sin, cos, pi, asin, sqrt, tan
from scipy.integrate import ode
from scipy.optimize import root
from scipy.interpolate import interp1d
from csgen.math_utils import cone_x, cone_y, cone_z
from csgen.compressible_flow import taylor_maccoll_stream_mach, theta_oblique, \
    p_pt, pt2_pt1_oblique, p2_p1_oblique
from csgen.stream_utils import Streamline
import pyvista as pv
import matplotlib.pyplot as plt
import csv

class BusemannField():
    # base class for Busemann flow fields
    def __init__(self, thetas, us, vs, rs, gamma, **kwargs):
        self.thetas = thetas
        self.us = us
        self.vs = vs
        self.rs = rs
        self.gamma = gamma
        self.mu = pi - self.thetas[0]

        self.xs = np.zeros(len(self.thetas))
        self.ys = np.nan * np.ones(len(self.thetas))
        self.zs = np.copy(self.ys)
        self.xyz_coords = np.nan * np.ones((len(self.thetas), 3))
        
        self.phis = np.full((len(self.thetas)), pi/2)
        self.polar_coords = np.nan * np.ones((len(self.thetas), 3))
        for i in range(len(self.thetas)):
            self.ys[i] = self.rs[i] * sin(self.thetas[i])
            self.zs[i] = self.rs[i] * cos(self.thetas[i])
            self.xyz_coords[i] = [self.xs[i], self.ys[i], self.zs[i]]
            self.polar_coords[i] = [self.rs[i], self.thetas[i], 
                                    self.phis[i]]
        
        self.Streamline = Streamline(polar_coords=self.polar_coords)
    
    def M3(self):
        # exit Mach number
        return Me(self.beta2, self.M2, self.gamma)
    
    def p2_p1(self):
        # compression ratio over terminating shock
        M1 = sqrt(self.us[0]**2 + self.vs[0]**2)
        return p_pt(self.M2, self.gamma) / p_pt(M1, self.gamma)
    
    def p3_p1(self):
        # compression ratio of entire flow field
        return self.p2_p1() * p2_p1_oblique(self.beta2, self.M2, self.gamma)
    
    def pt3_pt1(self):
        # total pressure ratio of entire flow field
        return 1 * pt2_pt1(self.beta2, self.M2, self.gamma)

    def plot(self, show_streamline=True, show_mach_wave=True, 
        show_exit_shock=True, show_plot=False, save_SVG=True, 
        file_name='buse_M2_beta2'):
        # create plot of streamline
        fig = plt.figure(figsize=(16, 9))
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.size": 20
        })
        ax = plt.axes()
        ax.plot([self.Streamline.zs[0], self.Streamline.zs[-1]], [0, 0], 'k-.', 
            label='Axis of Symmetry')
        
        if show_streamline == True:
            ax.plot(self.Streamline.zs, self.Streamline.ys, 'k-', 
                label='Busemann Contour')
        
        if show_mach_wave == True:
            ax.plot([self.Streamline.zs[0], 0], [self.Streamline.ys[0], 0], 
                'r--', label='Entrance Mach Wave')
        
        if show_exit_shock == True:
            ax.plot([0, self.Streamline.zs[-1]], [0, self.Streamline.ys[-1]], 
                'r-', label='Terminating Shock Wave')
        
        ax.set_xlabel('$z$')
        ax.set_ylabel('$y$')
        plt.axis('equal')
        plt.grid()
        plt.legend()
        
        if show_plot:
            plt.show()
        
        if save_SVG:
            fig.savefig(file_name + '.svg', bbox_inches='tight')

    def buse_surf(self, n_streams=100, save_VTK=True):
        # generate surface grid for 3D Busemann inlet
        phis_b = np.linspace(0, 2*pi, n_streams)
        buse_surf = np.nan * np.ones((len(phis_b), len(self.thetas), 3))
        for i in range(len(phis_b)):
            for j in range(len(self.thetas)):
                r_j = sqrt(self.xs[j]**2 + self.ys[j]**2 + self.zs[j]**2)
                buse_surf[i][j][0] = r_j * sin(self.thetas[j]) * cos(phis_b[i])
                buse_surf[i][j][1] = r_j * sin(self.thetas[j]) * sin(phis_b[i])
                buse_surf[i][j][2] = r_j * cos(self.thetas[j])
        
        # save as VTK, if desired
        if save_VTK == True:
            buse_grid = pv.StructuredGrid(buse_surf[:,:,0], buse_surf[:,:,1], 
            buse_surf[:,:,2])
            buse_grid.save("buse_surf.vtk")

        return buse_surf

    def mach_cone_surface(self, n_r=100, n_phi=100, save_VTK=True):
        # generate surface grid for 3D entrance Mach cone
        rs_mc = np.linspace(0, self.ys[0], n_r)
        phis_mc = np.linspace(0, 2*pi, n_phi)
        Rs_mc, Phis_mc = np.meshgrid(rs_mc, phis_mc)
        mc_xs = cone_x(Rs_mc, Phis_mc)
        mc_ys = cone_y(Rs_mc, Phis_mc)
        mc_zs = cone_z(mc_xs, mc_ys, self.mu, -1)
        mc_surf = np.nan * np.ones((len(rs_mc), len(phis_mc), 3))
        for i in range(len(rs_mc)):
            for j in range(len(phis_mc)):
                mc_surf[i][j] = [mc_xs[i][j], mc_ys[i][j], mc_zs[i][j]]
        
        # save as VTK, if desired
        if save_VTK == True:
            mc_grid = pv.StructuredGrid(mc_xs, mc_ys, mc_zs)
            mc_grid.save("mach_cone_surf.vtk")

        return mc_surf

    def term_shock_surface(self, n_r=100, n_phi=100, save_VTK=True):
        # generate surface grid for 3D terminating shock
        rs_ts = np.linspace(0, self.ys[-1], n_r)
        phis_ts = np.linspace(0, 2*pi, n_phi)
        Rs_ts, Phis_ts = np.meshgrid(rs_ts, phis_ts)
        ts_xs = cone_x(Rs_ts, Phis_ts)
        ts_ys = cone_y(Rs_ts, Phis_ts)
        ts_zs = cone_z(ts_xs, ts_ys, self.thetas[-1], 1)
        ts_surf = np.nan * np.ones((len(rs_ts), len(phis_ts), 3))
        for i in range(len(rs_ts)):
            for j in range(len(phis_ts)):
                ts_surf[i][j] = [ts_xs[i][j], ts_ys[i][j], ts_zs[i][j]]
        
        # save as VTK, if desired
        if save_VTK == True:
            ts_grid = pv.StructuredGrid(ts_xs, ts_ys, ts_zs)
            ts_grid.save("term_shock_surf.vtk")

        return ts_surf

#------------------------------------------------------------------------------#
#                          Integration Functions                               #
#------------------------------------------------------------------------------#
def sing_metric(u, v, theta):
    """
    Calculates the Taylor-Maccoll singularity metric when creating Busemann flow 
    fields.

    Arguments:
        u = tangential Mach number
        v = radial Mach number
        theta = angle from positive x-axis

    Returns:
        sing_metric: metric to determine if singularity has been reached;
                     sing_metric == 0 at singularity
    """
    return u + v / tan(theta)

def busemann_M2_beta2(design_vals, settings):
    """
    Creates a Busemann flow field characterised by M2 and beta2.

    Arguments:
        design_vals: 
            {'M2': Mach number at station 2
             'beta2': angle of terminating shock
             'r0': initial radius
             'gamma': ratio of specific heats}
        settings: 
            {'dtheta': integration step size for theta
             'max_steps': maximum number of integration steps
             'print_freq': printing frequency of integration info
             'interp_sing': if True, streamline will be interpolated at
                                     singularity point
             'verbosity': verbosity level (0 or 1)}
    
    Returns:
        BuseField: BusemannField object characterised by M2 and beta2
    """
    # unpack dictionaries
    M2 = design_vals['M2']
    beta2 = design_vals['beta2']
    r0 = design_vals.get('r0', 1.0)
    gamma = design_vals.get('gamma', 1.4)

    dtheta = settings.get('dtheta', 0.05*pi/180)
    max_steps = settings.get('max_steps', 10000)
    print_freq = settings.get('print_freq', 10)
    interp_sing = settings.get('interp_sing', True)
    verbosity = settings.get('verbosity', 1)

    # initial conditions
    a2 = theta_oblique(beta2, M2, gamma)
    theta2 = beta2 - a2
    u2 = M2 * cos(beta2)
    v2 = -M2 * sin(beta2)

    # integration settings
    r = ode(taylor_maccoll_stream_mach)
    r.set_integrator('dop853', nsteps=max_steps)
    ic = [theta2, u2, v2, r0]
    r.set_initial_value([ic[1], ic[2], ic[3]], ic[0])
    r.set_f_params(gamma)
    dt = dtheta
    
    # check singularity if singularity has been reached before integrating
    if sing_metric(r.y[0], r.y[1], r.t) >= 0:
        text = 'Singularity has been reached before integration. \n'
        text += f'theta2 = {ic[0] * 180/pi:.4} deg \n'
        text += f'[u, v] = [{ic[1]:.4}, {ic[2]:.4}]'
        raise ValueError(text)
    
    # intialise solution lists
    thetas = [ic[0]]
    us = [ic[1]]
    vs = [ic[2]]
    rs = [ic[3]]
    sms = [sing_metric(r.y[0], r.y[1], r.t)]
    
    # integrate Taylor-Maccoll equations
    if verbosity == 1:
        print('Integrating Taylor-Maccoll equations. \n')
    i = 0
    while r.successful() and sing_metric(r.y[0], r.y[1], r.t) < 0:
        r.integrate(r.t + dt)
        if verbosity == 1 and i % print_freq == 0:
            str_1 = f'Step={i} '
            str_2 = f'theta={r.t * 180/pi:.4} '
            str_3 = f'singularity metric={sing_metric(r.y[0], r.y[1], r.t):.4}'
            print(str_1 + str_2 + str_3)
        thetas.append(r.t)
        us.append(r.y[0])
        vs.append(r.y[1])
        rs.append(r.y[2])
        sms.append(sing_metric(r.y[0], r.y[1], r.t))
        i += 1

    # check if integration failed
    if r.successful() == False:
        raise Exception('Integration failed.')

    # use interpolation to find singularity point
    if interp_sing == True:
        # create cubic interpolation functions
        interp_thetas = interp1d(sms[-4:], thetas[-4:], kind='cubic')
        interp_us = interp1d(thetas[-4:], us[-4:], kind='cubic')
        interp_vs = interp1d(thetas[-4:], vs[-4:], kind='cubic')
        interp_rs = interp1d(thetas[-4:], rs[-4:], kind='cubic')

        # locate the theta value at the singularity
        theta_sing = float(interp_thetas(0))

        # interpolate u, v and r at theta_sing
        u_sing = float(interp_us(theta_sing))
        v_sing = float(interp_vs(theta_sing))
        r_sing = float(interp_rs(theta_sing))

        # replace last value of thetas, us, vs and rs with interpolated values
        thetas[-1] = theta_sing
        us[-1] = u_sing
        vs[-1] = v_sing
        rs[-1] = r_sing
        sms[-1] = 0
    else:
        # remove singularity from solution
        thetas.pop(-1)
        us.pop(-1)
        vs.pop(-1)
        rs.pop(-1)
        sms.pop(-1)

    # check if integration failed
    if r.successful == False:
        raise AssertionError('Integration failed.')

    # print solution at final step
    if verbosity == 1:
        print('\nIntegration was terminated due to singularity detection.')
        
        if interp_sing == True:
            print('Solution at interpolated singularity:')
        else:
            print('Solution at final step:')
        
        theta_f = thetas[-1] * 180/pi
        sm_f = sing_metric(us[-1], vs[-1], thetas[-1])
        mu_theory = asin(1/sqrt(us[-1]**2 + vs[-1]**2)) * 180/pi
        mu_calc = 180 - 180/pi * thetas[-1]

        print(f'theta = {theta_f:.4} deg')
        print(f'singularity metric = {sm_f:.4}')
        print(f'u = {us[-1]:.4}, v = {vs[-1]:.4}')
        print(f'theoretical Mach angle = {mu_theory:.4} deg')
        print(f'calculated entrance shock angle = {mu_calc:.4} deg')
    
    # return Busemann flow field object (reverse lists for increasing x)
    buse = BusemannField(thetas[::-1], us[::-1], vs[::-1], rs[::-1], gamma)
    buse.beta2 = beta2
    buse.M2 = M2
    buse.sms = sms[::-1]
    return buse

def busemann_M1_p3p1(design_vals, settings):
    """
    Creates a Busemann flow field characterised by M1 and p3/p1.

    Arguments:
        design_vals: 
            {'M1': Mach number at station 2
             'p3_p1': angle of terminating shock
             'r0': initial radius
             'gamma': ratio of specific heats}
        settings: 
            {'dtheta': integration step size for theta
             'beta2_guess': initial guess for beta2 [rad]
             'M2_guess': initial guess for M2
             'max_steps': maximum number of integration steps
             'print_freq': printing frequency of integration info
             'interp_sing': if True, streamline will be interpolated at
                                     singularity point
             'verbosity': verbosity level (0 or 1)}
    
    Returns:
        BuseField: BusemannField object characterised by M1 and p3/p1
    """
    # unpack dictionaries
    M1 = design_vals['M1']
    p3_p1 = design_vals['p3_p1']
    r0 = design_vals.get('r0', 1.0)
    gamma = design_vals.get('gamma', 1.4)

    dtheta = settings.get('dtheta', 0.05*pi/180)
    M2_guess = settings.get('M2_guess', 5.768)
    beta2_guess = settings.get('beta2_guess', 0.2410)
    max_steps = settings.get('max_steps', 10000)
    print_freq = settings.get('print_freq', 10)
    interp_sing = settings.get('interp_sing', True)
    verbosity = settings.get('verbosity', 1)

    settings_new = settings.copy()
    settings_new['verbosity'] = 0

    def res(x, M1, p3_p1):
        design_vals['M2'] = x[0]
        design_vals['beta2'] = x[1]

        buse = busemann_M2_beta2(design_vals, settings_new)
        M1_calc = sqrt(buse.us[0]**2 + buse.vs[0]**2)
        p3_p1_calc = buse.p3_p1()
        res_M1 = M1 - M1_calc
        res_p3p1 = p3_p1 - p3_p1_calc
        
        print(f'M1_residual={res_M1:.4} p3/p1_residual={res_p3p1:.4}')
        return [res_M1, res_p3p1] 
    
    if verbosity == 1:
        print('Using root finder to calculate M2 and beta2. \n')
    i = 0
    x_guess = [M2_guess, beta2_guess]
    sol = root(res, x_guess, args=(M1, p3_p1), method='hybr')
    if sol.success:
        print('\nRoot finder has converged. \n')
    else:
        raise AssertionError('Root finder failed to converge.')

    design_vals['M2'] = sol.x[0]
    design_vals['beta2'] = sol.x[1]
    buse = busemann_M2_beta2(design_vals, settings)
    return buse
