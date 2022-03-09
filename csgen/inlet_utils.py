"""
Utility functions for inlet design.

Author: Reece Otto 23/02/2022
"""
import numpy as np

def inlet_blend(inletA, inletB, z_cap, z_cowl, alpha):
    # check if inletA and inletB surface arrays have same shape
    if inletA.shape != inletB.shape:
        raise ValueError('Inlet surface grids contain different number of ' + \
        	'points.')

    def E(z, z_cap, z_cowl):
        #print(f'z - z_cap={z - z_cap}')
        return ((z - z_cap) / (z_cowl - z_cap))**alpha

    def f(z, sectionA_i, sectionB_i, z_cap, z_cowl, alpha):
        if sectionA_i < 0 or sectionB_i < 0:
        	return -abs(sectionA_i) ** (1 - E(z, z_cap, z_cowl)) * \
                    abs(sectionB_i) ** E(z, z_cap, z_cowl)
        else:
        	return abs(sectionA_i) ** (1 - E(z, z_cap, z_cowl)) * \
                   abs(sectionB_i) ** E(z, z_cap, z_cowl)

    inletA_xs = inletA[0]
    inletA_ys = inletA[1]
    inletA_zs = inletA[2]
    
    inletB_xs = inletB[0]
    inletB_ys = inletB[1]
    inletB_zs = inletB[2]
    
    """
    # check if inletA and inletB have same z coords
    if np.allclose(inletA_zs, inletB_zs) == False:
        raise ValueError('Inlet surfaces have different z \
                          coordinate arrays.')
    """
    inletC_xs = np.zeros(inletA_xs.shape)
    inletC_ys = np.zeros(inletA_ys.shape)
    inletC_zs = inletA_zs

    for i in range(len(inletA_xs)):
        for j in range(len(inletA_xs[0])):
            z_ij = inletC_zs[i][j]

            Ax_ij = inletA_xs[i][j]
            Bx_ij = inletB_xs[i][j]
            inletC_xs[i][j] = f(z_ij, Ax_ij, Bx_ij, z_cap, z_cowl, alpha)

            Ay_ij = inletA_ys[i][j]
            By_ij = inletB_ys[i][j]
            inletC_ys[i][j] = f(z_ij, Ay_ij, By_ij, z_cap, z_cowl, alpha)

    return [inletC_xs, inletC_ys, inletC_zs]