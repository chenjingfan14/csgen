"""
Minimizing surface area of cylinder with fixed volume of 375mL.

Author: Reece Otto 22/05/2022
"""
from pyoptsparse import SLSQP, Optimization, History
from math import pi

def surf_area(r, h):
	return 2*pi*r*(r + h)

def vol(r, h):
	return pi*h*r**2

# define objective function and constraints
def obj_func(design_vars):
	r = design_vars['radius']; h = design_vars['height']
	funcs = {}
	funcs['surf_area'] = surf_area(r, h)
	funcs['volume'] = vol(r, h)
	fail = False

	return funcs, fail

# optimization Object
opt_prob = Optimization("Cylinder Problem", obj_func)

# design variables
opt_prob.addVar('radius', 'c', lower=0.0, upper=0.1, value=0.05)
opt_prob.addVar('height', 'c', lower=0.0, upper=0.2, value=0.05)

# constraints
opt_prob.addCon('volume', lower=3.75E-4, upper=3.75E-4)

# objective
constant = 3
opt_prob.addObj('surf_area')

# print optimization problem information
#print(opt_prob)

# define optimization routine
opt_options = {'IPRINT': 1, 'ACC':1E-08}
opt = SLSQP(options=opt_options)

# solve 
sol = opt(opt_prob, sens='FD', storeHistory='hist')

# print info
print(f'Minimum surface area = {sol.fStar[0]:.4} m^2')
print(f"Cylinder radius = {1000*sol.xStar['radius']:.4} mm")
print(f"Cylinder height = {1000*sol.xStar['height']:.4} mm")