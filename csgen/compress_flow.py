"""
Compressible flow relations.

Author: Reece Otto 29/03/2022
"""
from math import sin, tan, atan, sqrt, pi
from scipy.optimize import newton
#------------------------------------------------------------------------------#
#                       Compressible Flow Equations                            #
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#                         Normal Shock Relations                               #
#------------------------------------------------------------------------------#
def rho2_rho1_normal(M, gamma):
    """
    Calculates density ratio over a normal shockwave.

    Arguments:
        M: Mach number upstream of shock
        gamma: ratio of specific heats

    Returns:
        rho2/rho1: density ratio
    """
    return (gamma+1) * M*M / ((gamma-1) * M*M + 2)

def V2_V1_normal(M, gamma):
    """
    Calculates velocity ratio over a normal shockwave.

    Arguments:
        M: Mach number upstream of shock
        gamma: ratio of specific heats

    Returns:
        V2/V1: velocity ratio
    """
    return ((gamma-1) * M*M + 2) / ((gamma+1) * M*M)

#------------------------------------------------------------------------------#
#                        Oblique Shock Relations                               #
#------------------------------------------------------------------------------#
def theta_oblique(beta, M, gamma):
    """
    Calculates theta from theta-beta-M relation.
    
    Arguments:
        beta: oblique shock angle
        M: Mach number upstream of shock
        gamma: ratio of specific heats
    
    Returns:
        theta: flow deflection angle

    """
    k = M * sin(beta)
    numer = 2 * (k*k - 1) / tan(beta)
    denom = (gamma+1) * M*M - 2*(k*k - 1)
    return atan(numer / denom)

def beta_oblique(theta, M, gamma):
    """
    Calculates beta from theta-beta-M relation.
    
    Arguments:
        theta: flow deflection angle
        M: Mach number upstream of shock
        gamma: ratio of specific heats
    
    Returns:
        beta: oblique shock angle

    """
    def res(beta, theta, M, gamma):
        k = M * sin(beta)
        term1 = 1 / tan(theta)
        term2 = tan(beta) * ((gamma+1) * M*M / (2*(k*k-1)) - 1)
        return term1 - term2
    
    beta_guess = 10 * pi / 180
    return newton(res, beta_guess, args=(theta, M, gamma), maxiter=200)

def M2_oblique(beta, delta, M1, gamma):
    """
    Calculates Mach number downstream of oblique shockwave.
    
    Arguments:
        beta: oblique shock angle
        M1: Mach number upstream of shock
        gamma: ratio of specific heats
    
    Returns:
        M2: Mach number downstream of oblique shock
    """
    k = M1 * sin(beta)
    numer = ((gamma-1) * k*k + 2) / (sin(beta-delta))**2
    denom = 2*gamma*k*k - gamma + 1
    return sqrt(numer/denom)

def p2_p1_oblique(beta, M, gamma):
    """
    Calculates static pressure ratio over oblique shockwave.
    
    Arguments:
        beta: shock angle
        M: up-stream Mach number
        gamma: ratio of specific heats
    
    Returns:
        p2/p1: static pressure ratio over oblique shock
    """
    return (2*gamma * (M*sin(beta))**2 - gamma+1) / (gamma+1)

def pt2_pt1_oblique(beta, M, gamma):
    """
    Calculates total pressure ratio over oblique shockwave.
    
    Arguments:
        beta: shock angle
        M: up-stream Mach number
        gamma: ratio of specific heats
    
    Returns:
        pt2/pt1: temperature ratio over oblique shock
    """
    k = M * sin(beta)
    factor1 = (gamma+1) * k*k / ((gamma-1) * k*k + 2)
    factor2 = (gamma+1) / (2*gamma*k*k - gamma+1)
    return factor1**(gamma / (gamma-1)) * factor2**(1 / (gamma-1))

def T2_T1_oblique(beta, M, gamma):
    """
    Calculates temperature ratio over oblique shockwave.
    
    Arguments:
        beta: shock angle
        M: up-stream Mach number
        gamma: ratio of specific heats
    
    Returns:
        T2/T1: temperature ratio over oblique shock
    """
    k = M * sin(beta)
    numer = (2*gamma * (k*k - gamma+1)) * ((gamma-1) * k*k + 2)
    denom = k*k * (gamma+1)**2
    return numer/denom

#------------------------------------------------------------------------------#
#                       Isentropic Flow Relations                              #
#------------------------------------------------------------------------------#
def p_pt(M, gamma):
    """
    Calculates the ratio of static pressure to total pressure.
    
    Arguments:
        M: Mach number
        gamma: ratio of specific heats
    
    Returns:
        p/pt: ratio of static pressure to total pressure
    """
    return (1 + (gamma-1) * M*M/2)**(-gamma / (gamma-1)) 

def T_Tt(M, gamma):
    """
    Calculates the ratio of temperature to total temperature.
    
    Arguments:
        M: Mach number
        gamma: ratio of specific heats
    
    Returns:
        T/Tt: ratio of temperature to total temperature
    """
    return (1 + (gamma-1) * M*M/2)**(-1)

def rho_rhot(M, gamma):
    """
    Calculates the ratio of density to total density.
    
    Arguments:
        M: Mach number
        gamma: ratio of specific heats
    
    Returns:
        rho/rhot: ratio of density to total density
    """
    return (1 + (gamma-1) * M*M/2)**(-1 / (gamma-1))

def p_q(M, gamma):
    """
    Calculates the ratio of static pressure to dynamic pressure.
    
    Arguments:
        M: Mach number
        gamma: ratio of specific heats
    
    Returns:
        p/q: ratio of static pressure to dynamic pressure
    """
    return 2 / (gamma*M*M)

#------------------------------------------------------------------------------#
#                        Conical Flow Relations                                #
#------------------------------------------------------------------------------#
def taylor_maccoll_vel(theta, y, gamma):
    """
    Taylor-Maccoll equations in polar form.
    Note: equations are expressed in terms of radial and angular velocities that
    are nondimensionalised by their maximum value: sqrt(2 * total enthalpy)
    
    Arguments:
        theta: angle measured counterclockwise from positive x direction
        y: [U, V] = [radial velocity, angular velocity]
        gamma: ratio of specific heats
    
    Returns:
        [dU/dtheta, dV/dtheta, dr/dtheta]: theta derivatives
    """
    U = y[0]
    V = y[1]
    
    dU_dtheta = V
    dV_dtheta = (U * V*V - (gamma-1)*(1 - U*U - V*V)*(2*U + V/tan(theta))/2) / \
               ((gamma-1)*(1 - U*U - V*V)/2 - V*V)
    return [dU_dtheta, dV_dtheta]

def taylor_maccoll_mach(theta, y, gamma):
    """
    Taylor-Maccoll equations in polar form.
    Note: equations are expressed in terms of radial and angular Mach numbers
          rather than velocities.
    
    Returns:
        theta: angle measured counterclockwise from positive x direction
        y: [u, v] = [radial Mach number, angular Mach number]     
        gamma: ratio of specific heats
    """
    u = y[0]
    v = y[1]
    
    du_dtheta = v + (gamma-1) * u*v* (u + v/tan(theta)) / (2 * (v*v - 1))     
    dv_dtheta = -u + (1 + (gamma-1) * v*v/2) * (u + v/tan(theta)) / \
                (v*v - 1)
    return [du_dtheta, dv_dtheta]

def taylor_maccoll_stream_mach(theta, y, gamma):
    """
    Taylor-Maccoll equations and streamline equation in polar form.
    Note: equations are expressed in terms of radial and angular Mach numbers
          rather than velocities.
    
    Returns:
        theta: angle measured counterclockwise from positive x direction
        y: [u, v, r] = [radial Mach number, angular Mach number, 
                        radius from shock focal point]     
        gamma: ratio of specific heats
    """
    u = y[0]
    v = y[1]
    r = y[2]
    
    du_dtheta = v + (gamma-1) * u*v* (u + v/tan(theta)) / (2*(v*v - 1))     
    dv_dtheta = -u + (1 + (gamma-1) * v*v/2) * (u + v/tan(theta)) / \
                (v*v - 1)
    dr_dtheta = r * u/v
    return [du_dtheta, dv_dtheta, dr_dtheta]