"""
Utility functions for inlet design.

Author: Reece Otto 23/02/2022
"""
import numpy as np

def inlet_blend(inletA, inletB, alpha):
    # check if inletA and inletB surface arrays have same shape
    if inletA.shape != inletB.shape:
        raise ValueError('Inlet surface grids contain different number of ' + \
        	'points.')

    def E(z, z_start, z_end):
        return ((z - z_start) / (z_end - z_start))**alpha

    def f(z, coord_A_ij, coord_B_ij, z_start, z_end, alpha):
        if coord_A_ij < 0 or coord_B_ij < 0:
            return -abs(coord_A_ij) ** (1 - E(z, z_start, z_end)) * \
                   abs(coord_B_ij) ** E(z, z_start, z_end)
        else:
            return abs(coord_A_ij) ** (1 - E(z, z_start, z_end)) * \
                   abs(coord_B_ij) ** E(z, z_start, z_end)

    inletA_xs = inletA[:,:,0]
    inletA_ys = inletA[:,:,1]
    inletA_zs = inletA[:,:,2]

    inletB_xs = inletB[:,:,0]
    inletB_ys = inletB[:,:,1]
    inletB_zs = inletB[:,:,2]

    inletC_xs = np.zeros(inletA_xs.shape)
    inletC_ys = np.zeros(inletA_ys.shape)
    inletC_zs = np.zeros(inletA_zs.shape)

    n_streams = len(inletA_zs)
    n_z = len(inletA_zs[0])

    for i in range(n_streams):
        # calculate list of z values for streamline i
        z_min_i = np.amin(inletA_zs[i])
        z_max_i = np.amax(inletB_zs[i])
        zs_i = np.linspace(z_max_i, z_min_i, n_z)
        for j in range(n_z):
            z_ij = zs_i[j]
            inletC_zs[i][j] = z_ij

            A_x_ij = inletA_xs[i][j]
            B_x_ij = inletB_xs[i][j]
            inletC_xs[i][j] = f(z_ij, A_x_ij, B_x_ij, z_min_i, z_max_i, alpha)

            A_y_ij = inletA_ys[i][j]
            B_y_ij = inletB_ys[i][j]
            inletC_ys[i][j] = f(z_ij, A_y_ij, B_y_ij, z_min_i, z_max_i, alpha)

    return [inletC_xs, inletC_ys, inletC_zs]