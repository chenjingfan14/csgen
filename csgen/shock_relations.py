"""
Rankine-Hugoniot equations.

Author: Reece Otto 03/11/2021
"""
from math import sin, tan, atan, sqrt, pi
from scipy.optimize import newton

# compressible flow relations
def sonic_speed(T, R, gamma):
    return sqrt(R * gamma * T)

def air_sonic_speed(T):
    return sonic_speed(T, 287.1, 1.4)

# normal shock relations
def rho2_rho1_normal(M, gamma):
    return (gamma + 1) * M*M / ((gamma - 1) * M*M + 2)

def V2_V1_normal(M, gamma):
    return ((gamma - 1) * M*M + 2) / ((gamma + 1) * M*M)

# oblique shock relations
def theta_oblique(beta, M, gamma):
    """
    Calculates theta from theta-beta-M relation.
    
    Inputs:
        beta = oblique shock angle
        M = Mach number upstream of shock
        gamma = ratio of specific heats
    
    Output:
        theta = flow deflection angle

    """
    k = M * sin(beta)
    numer = 2 * (k*k - 1) / tan(beta)
    denom = (gamma + 1) * M*M - 2 * (k*k - 1)
    return atan(numer / denom)

def beta_oblique(theta, M, gamma):
    """
    Calculates beta from theta-beta-M relation.
    
    Inputs:
        theta = flow deflection angle
        M = Mach number upstream of shock
        gamma = ratio of specific heats
    
    Output:
        beta = oblique shock angle

    """
    def res(beta, theta, M, gamma):
        k = M * sin(beta)
        term1 = 1 / tan(theta)
        term2 = tan(beta) * ((gamma + 1) * M*M / (2 * (k*k - 1)) - 1)
        return term1 - term2
    
    beta_guess = 10 * pi / 180
    return newton(res, beta_guess, args=(theta, M, gamma), maxiter=200)

def M2_oblique(beta, delta, M1, gamma):
    """
    TODO: does this always hold true for non Busemann flow?
    Calculates Mach number downstream of oblique shockwave.
    
    Inputs:
        beta = oblique shock angle
        M1 = Mach number upstream of shock
        gamma = ratio of specific heats
    
    Output:
        M2 = Mach number downstream of oblique shock
    """
    k = M1 * sin(beta)
    """
    numer = (gamma + 1)**2 * M1*M1 * k*k - 4 * (k*k - 1) * \
            (gamma * k*k + 1)
    denom = (2 * gamma * k*k - gamma + 1) * ((gamma - 1) * k*k + 2)
    """
    numer = ((gamma-1) * k*k + 2) / (sin(beta - delta))**2
    denom = 2 * gamma * k*k - gamma + 1
    return sqrt(numer / denom)

def p2_p1_oblique(beta, M, gamma):
    """
    Calculates static pressure ratio over oblique shockwave.
    
    Inputs:
        beta = shock angle
        M = up-stream Mach number
        gamma = ratio of specific heats
    
    Output:
        p2/p1 = static pressure ratio over oblique shock
    """
    return (2 * gamma * (M * sin(beta))**2 - gamma + 1) / (gamma + 1)

def pt2_pt1_oblique(beta, M, gamma):
    """
    Calculates total pressure ratio over oblique shockwave.
    
    Inputs:
        beta = shock angle
        M = up-stream Mach number
        gamma = ratio of specific heats
    
    Output:
        pt2/pt1 = temperature ratio over oblique shock
    """
    k = M * sin(beta)
    factor1 = (gamma + 1) * k*k / ((gamma - 1) * k*k + 2)
    factor2 = (gamma + 1) / (2 * gamma * k*k - gamma + 1)
    return factor1**(gamma / (gamma - 1)) * factor2**(1 / (gamma - 1))

def T2_T1_oblique(beta, M, gamma):
    """
    Calculates temperature ratio over oblique shockwave.
    
    Inputs:
        beta = shock angle
        M = up-stream Mach number
        gamma = ratio of specific heats
    
    Output:
        T2/T1 = temperature ratio over oblique shock
    """
    k = M * sin(beta)
    numer = (2 * gamma * (k*k - gamma + 1)) * ((gamma - 1) * k*k + 2)
    denom = k*k * (gamma + 1)**2
    return numer / denom

