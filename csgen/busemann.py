"""
Functions for creating Busemann inlet contours.

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
from csgen.taylor_maccoll import tm_stream_mach
from csgen.stream_utils import PolarStreamline
from csgen.shock_relations import theta_oblique                   
from csgen.isentropic_flow import p_pt
from csgen.shock_relations import pt2_pt1_oblique, p2_p1_oblique
import matplotlib.pyplot as plt

def cone_x(r, phi):
    return r * np.cos(phi)
    
def cone_y(r, phi):
    return r * np.sin(phi)
    
def cone_z(x, y, theta, sign):
    a = tan(theta)
    return sign * np.sqrt((x*x + y*y) / (a*a))

class BusemannField():
    # base class for Busemann flow fields
    def __init__(self, thetas, us, vs, rs, gamma, **kwargs):
        self.thetas = thetas
        self.us = us
        self.vs = vs
        self.rs = rs
        self.gamma = gamma
        self.mu = pi - self.thetas[-1]
    
    def streamline(self, phi=pi/2):
        # construct phi array
        phis = np.full((len(self.rs)), phi)

        # return streamline with coords that are in order of increasing x
        return PolarStreamline(self.rs[::-1], self.thetas[::-1], phis)
    
    def surface(self, n_streams=100):
        # generate surface grid for 3D Busemann inlet
        phis_b = np.linspace(pi/2, 5*pi/2, n_streams)
        stream = self.streamline()
        buse_xs = np.zeros((len(phis_b), len(stream.xs)))
        buse_ys = np.zeros((len(phis_b), len(stream.xs)))
        buse_zs = np.zeros((len(phis_b), len(stream.xs)))
        for i in range(len(phis_b)):
            for j in range(len(stream.xs)):
                r_j = sqrt(stream.xs[j]**2 + stream.ys[j]**2 + stream.zs[j]**2)
                buse_xs[i][j] = r_j * sin(stream.thetas[j]) * cos(phis_b[i])
                buse_ys[i][j] = r_j * sin(stream.thetas[j]) * sin(phis_b[i])
                buse_zs[i][j] = r_j * cos(stream.thetas[j])
        return [buse_xs, buse_ys, buse_zs]

    def mach_cone_surface(self, n_r=100, n_phi=100):
        # generate surface grid for 3D entrance Mach cone
        stream = self.streamline()
        rs_mc = np.linspace(0, stream.ys[0], n_r)
        phis_mc = np.linspace(0, 2*pi, n_phi)
        Rs_mc, Phis_mc = np.meshgrid(rs_mc, phis_mc)
        mc_xs = cone_x(Rs_mc, Phis_mc)
        mc_ys = cone_y(Rs_mc, Phis_mc)
        mc_zs = cone_z(mc_xs, mc_ys, self.mu, -1)
        return [mc_xs, mc_ys, mc_zs]

    def term_shock_surface(self, n_r=100, n_phi=100):
        # generate surface grid for 3D terminating shock
        stream = self.streamline()
        rs_ts = np.linspace(0, stream.ys[-1], n_r)
        phis_ts = np.linspace(0, 2*pi, n_phi)
        Rs_ts, Phis_ts = np.meshgrid(rs_ts, phis_ts)
        ts_xs = cone_x(Rs_ts, Phis_ts)
        ts_ys = cone_y(Rs_ts, Phis_ts)
        ts_zs = cone_z(ts_xs, ts_ys, self.thetas[0], 1)
        return [ts_xs, ts_ys, ts_zs]

    def M3(self):
        # exit Mach number
        return Me(self.beta2, self.M2, self.gamma)
    
    def p2p1(self):
        # compression ratio over terminating shock
        M1 = sqrt(self.us[-1]**2 + self.vs[-1]**2)
        return p_pt(self.M2, self.gamma) / p_pt(M1, self.gamma)
    
    def p3p1(self):
        # compression ratio of entire flow field
        return self.p2p1() * p2_p1_oblique(self.beta2, self.M2, self.gamma)
    
    def pt3pt1(self):
        # total pressure ratio of entire flow field
        return 1 * pt2_pt1(self.beta2, self.M2, self.gamma)

def sing_metric(u, v, theta):
    """
    Calculates the Taylor-Maccoll singularity metric when creating Busemann flow 
    fields.

    Arguments:
        u = tangential Mach number
        v = radial Mach number
        theta = angle from positive x-axis
    """
    return u + v / tan(theta)

def busemann_M2_beta2(M2, beta2, gamma=1.4, dtheta=0.001*pi/180, r0=1, 
    n_steps=10000, print_freq=100, interp_sing=True, verbosity=1):
    """
    Creates a Busemann flow field characterised by M2 and beta2.

    Arguments:
        M2 = Mach number at station 2
        beta2 = angle of terminating shock
    
    Keyword arguments:
        gamma = ratio of specific heats
        h = throat height
        dtheta = step size for theta when integrating
        verbosity = set to 1 for full terminal output
        print_freq = integration information is printed on steps that are a
                     multiple of print_freq
    """
    # initial condition
    a2 = theta_oblique(beta2, M2, gamma)
    theta2 = beta2 - a2
    u2 = M2 * cos(beta2)
    v2 = -M2 * sin(beta2)

    # integration settings
    r = ode(tm_stream_mach).set_integrator('dop853', nsteps=n_steps)
    ic = [theta2, u2, v2, r0]
    r.set_initial_value([ic[1], ic[2], ic[3]], ic[0])
    r.set_f_params(gamma)
    dt = dtheta
    
    # check singularity if singularity has been reached before integrating
    if sing_metric(r.y[0], r.y[1], r.t) >= 0:
        text = 'Singularity has been reached before integration. \n'
        text += f'theta2 = {ic[0] * 180/pi:.4} deg \n'
        text += f'[u, v] = {[ic[1], ic[2]]}'
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
        theta_sing = interp_thetas(0)

        # interpolate u, v and r at theta_sing
        u_sing = interp_us(theta_sing)
        v_sing = interp_vs(theta_sing)
        r_sing = interp_rs(theta_sing)

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

    # print integration termination statement
    if r.successful == False:
        raise AssertionError('Integration failed.')

    if verbosity == 1:
        print('\nIntegration was terminated.')
        reason = 'Reason: '
        # print reason for integration termination
        if sing_metric(r.y[0], r.y[1], r.t) >= 0:
            print(reason + 'Taylor-Maccoll singularity detected. \n')
        else:
            print(reason + 'Unknown. \n')
        
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
    
    # return Busemann flow field object
    buse = BusemannField(thetas, us, vs, rs, gamma)
    buse.beta2 = beta2
    buse.M2 = M2
    buse.sms = sms
    return buse

def busemann_M1_p3p1(M1, p3p1, gamma=1.4, dtheta=0.001*pi/180, n_steps=10000,
    print_freq=100, interp_sing=True, verbosity=1, beta2_guess=0.2410, 
    M2_guess=5.768):
    # provide guess for beta2 and M2
    # beta2 = 14.5 deg (0.2410 rad), M2 = 5.5 is good for M = 10, p3 = 50kPa
    x_guess = [beta2_guess, M2_guess]
    
    def res(x, M1, p3p1):
        beta2 = x[0]
        M2 = x[1]

        buse = busemann_M2_beta2(M2, beta2, gamma, dtheta=dtheta, 
            n_steps=n_steps, print_freq=print_freq, interp_sing=interp_sing, 
            verbosity=0)
        M1_calc = sqrt(buse.us[-1]**2 + buse.vs[-1]**2)
        p3p1_calc = buse.p3p1()
        res_M1 = M1 - M1_calc
        res_p3p1 = p3p1 - p3p1_calc
        
        if verbosity == 1:
            print(f'M1_residual={res_M1:.4} p3/p1_residual={res_p3p1:.4}')
        return [res_M1, res_p3p1] 
    
    if verbosity == 1:
        print('Using root finder to calculate M2 and beta2. \n')
    i = 0
    sol = root(res, x_guess, args=(M1, p3p1), method='hybr')
    if sol.success:
        print('\nRoot finder has converged. \n')
    else:
        raise ValueError('Root finder failed to converge.')
    
    beta2 = sol.x[0]
    M2 = sol.x[1]
    buse = busemann_M2_beta2(M2, beta2, gamma, dtheta=dtheta, n_steps=n_steps, 
        print_freq=print_freq, interp_sing=interp_sing, verbosity=verbosity)
    return buse
