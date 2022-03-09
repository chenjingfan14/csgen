"""
Isentropic flow relations.

Author: Reece Otto 03/11/2021
"""

def p_pt(M, gamma):
    """
    Calculates the ratio of static pressure to total pressure.
    
    Inputs:
        M = up-stream Mach number
        gamma = ratio of specific heats
    
    Outputs:
        p/pt = ratio of static pressure to total pressure
    """
    return (1 + (gamma - 1) * M*M / 2)**(-gamma / (gamma - 1)) 

def T_Tt(M, gamma):
    """
    Calculates the ratio of temperature to total temperature.
    
    Inputs:
        M = up-stream Mach number
        gamma = ratio of specific heats
    
    Outputs:
        T/Tt = ratio of temperature to total temperature
    """
    return (1 + (gamma - 1) * M*M / 2)**(-1)

def rho_rhot(M, gamma):
    """
    Calculates the ratio of density to total density.
    
    Inputs:
        M = up-stream Mach number
        gamma = ratio of specific heats
    
    Outputs:
        rho/rhot = ratio of density to total density
    """
    return (1 + (gamma - 1) * M*M / 2)**(-1 / (gamma - 1))

def p_q(M, gamma):
    """
    Calculates the ratio of static pressure to dynamic pressure.
    
    Inputs:
        M = up-stream Mach number
        gamma = ratio of specific heats
    
    Outputs:
        p/q = ratio of static pressure to dynamic pressure
    """
    return 2 / (gamma * M*M)