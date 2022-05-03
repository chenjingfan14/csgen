"""
Mathematical utility functions.

Author: Reece Otto 29/03/2022
"""
import numpy as np
from math import sqrt, sin, cos, tan, atan2

def cone_x(r, phi):
    return r * np.cos(phi)
    
def cone_y(r, phi):
    return r * np.sin(phi)
    
def cone_z(x, y, theta, sign):
    a = tan(theta)
    return sign * np.sqrt((x*x + y*y)/(a*a))

