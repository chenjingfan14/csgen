"""
Routines used for generating conical waveriders.

Author: Reece Otto 29/03/2022
"""
from csgen.atmosphere import atmos_interp
from csgen.conical_field import conical_M0_thetac
from math import pi

def conical_stream(design_param, settings):
	

	
	# calculate remaining free-stream properties from US standard atmopshere 1976
	p0 = 2 * q0 / (gamma * M0*M0)                    # static pressure (Pa)
	T0 = atmos_interp(p0, 'Pressure', 'Temperature') # temperature (K)
	a0 = atmos_interp(p0, 'Pressure', 'Sonic Speed') # sonic speed (m/s)
	V0 = M0 * a0                                     # flight speed (m/s)

	# generate flow field
	field = conical_M0_thetac(M0, thetac, gamma, dtheta, beta_guess=beta_guess, 
		n_steps=n_steps, max_iter=max_iter, tol=tol, interp_sing=interp_sing, 
		verbosity=verbosity, print_freq=print_freq)
	stream_coords = field.streamline(scale=[1, -1, 1], L_field=2*z_base)
	return stream_coords
