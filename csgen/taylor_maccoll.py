"""
Tools for solving the Taylor-Maccoll equations

Author: Reece Otto 03/11/2021
"""
from math import tan, sin, cos, sqrt, pi
from csgen.shock_relations import theta_oblique

def taylor_maccoll_vel(theta, y, gamma, h):
    """
    Taylor-Maccoll equations in polar form.
    Note: equations are expressed in terms of radial and angular velocities that
    are nondimensionalised by their maximum value: sqrt(2 * h0)
    
    Arguments:
        theta = angle measured counterclockwise from positive x direction
        y = [U, V] = [radial velocity, angular velocity]
        gamma = ratio of specific heats
    
    Output:
        [dU/dtheta, dV/dtheta, dr/dtheta] = theta derivatives
    """
    U = y[0]
    V = y[1]
    
    dU_dtheta = V
    dV_dtheta = (U * V*V - (gamma - 1)*(1 - U*U - V*V)*(2*U + V/tan(theta))/2) /\
               ((gamma - 1)*(1 - U*U - V*V)/2 - V*V)
    return [dU_dtheta, dV_dtheta]

def taylor_maccoll_mach(theta, y, gamma):
    """
    Taylor-Maccoll equations in polar form.
    Note: equations are expressed in terms of radial and angular Mach numbers
          rather than velocities.
    
    Arguments:
        theta = angle measured counterclockwise from positive x direction
        y = [u, v] = [radial Mach number (U/a), angular Mach number (V/a)]     
        gamma = ratio of specific heats
    """
    u = y[0]
    v = y[1]
    
    du_dtheta = v + (gamma - 1) * u * v * (u + v / tan(theta)) / (2 * (v*v - 1))     
    dv_dtheta = -u + (1 + (gamma - 1) * v*v / 2) * (u + v / tan(theta)) \
               / (v*v - 1)
    return [du_dtheta, dv_dtheta]

def tm_stream_mach(theta, y, gamma):
    """
    Taylor-Maccoll equations and streamline equation in polar form.
    Note: equations are expressed in terms of radial and angular Mach numbers
          rather than velocities.
    
    Arguments:
        theta = angle measured counterclockwise from positive x direction
        y = [u, v, r] = [radial Mach number (U/a), angular Mach number (V/a), 
                         radius from shock focal point]     
        gamma = ratio of specific heats
    """
    u = y[0]
    v = y[1]
    r = y[2]
    
    du_dtheta = v + (gamma - 1) * u * v * (u + v / tan(theta)) / (2 * (v*v - 1))     
    dv_dtheta = -u + (1 + (gamma - 1) * v*v / 2) * (u + v / tan(theta)) \
               / (v*v - 1)
    dr_dtheta = r * u / v
    return [du_dtheta, dv_dtheta, dr_dtheta]