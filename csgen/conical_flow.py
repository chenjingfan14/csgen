"""
Tools for creating conical flow fields.

Author: Reece Otto 13/12/2021
"""
from csgen.compress_flow import taylor_maccoll_mach, theta_oblique, \
    M2_oblique, beta_oblique, p2_p1_oblique, T2_T1_oblique, p_pt, T_Tt
from csgen.stream_utils import Streamline
from csgen.math_utils import cone_x, cone_y, cone_z
from scipy.integrate import ode
from scipy.optimize import root
from scipy.interpolate import interp1d
from math import pi, cos, sin, tan, sqrt, atan, isclose
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from vtkmodules.vtkCommonDataModel import vtkStructuredGrid
from vtkmodules.vtkCommonCore import vtkPoints, vtkDoubleArray
from vtkmodules.vtkIOXML import vtkXMLStructuredGridWriter

class ConicalField():
    # base class for conical flow fields
    def __init__(self, thetas, us, vs, M0, beta, thetac, gamma):
        self.thetas = thetas
        self.us = us
        self.vs = vs
        self.M0 = M0
        self.beta = beta
        self.thetac = thetac
        self.gamma = gamma

    def u(self, theta):
        # check if given theta value is in range of valid values
        if isclose(theta, self.thetas[-1], abs_tol=1.0E-15):
            theta = self.thetas[-1]
        elif isclose(theta, self.thetas[0], abs_tol=1.0E-15):
            theta = self.thetas[0]
        
        if theta > self.thetas[0] or theta < self.thetas[-1]:
            raise AssertionError('Invalid theta value given.')

        if theta == self.thetas[0]:
            return self.us[0]
        elif theta == self.thetas[-1]:
            return self.us[-1]
        else:
            # find index of value in thetas that is <= given theta value
            ind = int(np.argwhere(np.array(self.thetas) > theta)[-1][0])

            # create cubic interpolation functions
            thetas_bnd = self.thetas[ind:ind+2]
            interp_us = interp1d(thetas_bnd, self.us[ind:ind+2])

            # calculate u and v at given theta value
            return interp_us(theta)

    def v(self, theta):
        # check if theta is close to boundaries
        if isclose(theta, self.thetas[-1], abs_tol=1.0E-15):
            theta = self.thetas[-1]
        elif isclose(theta, self.thetas[0], abs_tol=1.0E-15):
            theta = self.thetas[0]

        if theta > self.thetas[0] or theta < self.thetas[-1]:
            raise AssertionError('Invalid theta value given.')

        if theta == self.thetas[0]:
            return self.vs[0]
        elif theta == self.thetas[-1]:
            return self.vs[-1]
        else:
            # find index of value in thetas that is <= given theta value
            ind = int(np.argwhere(np.array(self.thetas) > theta)[-1][0])

            # create cubic interpolation functions
            thetas_bnd = self.thetas[ind:ind+2]
            interp_vs = interp1d(thetas_bnd, self.vs[ind:ind+2])

            # calculate u and v at given theta value
            return interp_vs(theta)

    def M(self, theta):
        if isclose(theta, self.thetas[-1], abs_tol=1.0E-15):
            theta = self.thetas[-1]
        elif isclose(theta, self.thetas[0], abs_tol=1.0E-15):
            theta = self.thetas[0]

        if theta > self.thetas[0] or theta < self.thetas[-1]:
            raise AssertionError('Invalid theta value given.')

        if theta == self.thetas[0]:
            return sqrt(self.us[0]**2 + self.vs[0]**2)
        elif theta == self.thetas[-1]:
            return sqrt(self.us[-1]**2 + self.vs[-1]**2)
        else:
            # find index of value in thetas that is <= given theta value
            ind = int(np.argwhere(np.array(self.thetas) > theta)[-1][0])

            # create cubic interpolation functions
            thetas_bnd = self.thetas[ind:ind+2]
            interp_us = interp1d(thetas_bnd, self.us[ind:ind+2])
            interp_vs = interp1d(thetas_bnd, self.vs[ind:ind+2])

            # calculate u and v at given theta value
            u = interp_us(theta)
            v = interp_vs(theta)

            # calculate Mach number
            return sqrt(u*u + v*v)

    def p(self, theta, p0):
        # find Mach number at given theta value
        Ma = self.M(theta)

        # calculate all relevant pressure ratios
        p_pt1 = p_pt(Ma, self.gamma)
        M1 = sqrt(self.us[0]**2 + self.vs[0]**2)
        pt1_p1 = 1 / p_pt(M1, self.gamma)
        p1_p0 = p2_p1_oblique(self.beta, self.M0, self.gamma)

        # calculate pressure
        return p0 * p_pt1 * pt1_p1 * p1_p0

    def T(self, theta, T0):
        # find Mach number at given theta value
        Ma = self.M(theta)

        # calculate all relevant temperature ratios
        T_Tt1 = T_Tt(Ma, self.gamma)
        M1 = sqrt(self.us[0]**2 + self.vs[0]**2)
        Tt1_T1 = 1 / T_Tt(M1, self.gamma)
        T1_T0 = T2_T1_oblique(self.beta, self.M0, self.gamma)

        # calculate temperature
        return T0 * T_Tt1 * Tt1_T1 * T1_T0

    def Streamline(self, config):
        # unpack dictionaries
        L_field = config['length']
        r0 = config.get('init_radius', 1.0)
        max_steps = config.get('max_steps', 10000)
        print_freq = config.get('print_freq', 10)
        verbosity = config.get('verbosity', 1)

        def stream_eqn(theta, r):
            # polar form of the streamline equation
            global i
            u = self.us[i]
            v = self.vs[i]
            return r * u / v

        # integration settings 
        r = ode(stream_eqn).set_integrator('DOP853', nsteps=max_steps)
        r.set_initial_value([r0], self.thetas[0])
        rs = [r0]

        # begin integration
        if verbosity == 1:
            print('Integrating streamline equation.')
        global i
        i = 0
        while r.successful() and i < len(self.thetas)-1 and \
        r.y[0]*cos(r.t) < L_field:
            r.integrate(self.thetas[i+1])
            if verbosity == 1 and i % print_freq == 0:
                str_1 = f'Step={i} '
                str_2 = f'theta={r.t * 180/pi:.4} '
                str_3 = f'r={r.y[0]:.4}'
                print(str_1 + str_2 + str_3)
            rs.append(r.y[0])
            i += 1
        print()
        
        # calculate Cartesian coordinates of streamline
        coords = np.nan * np.ones((len(rs), 3))
        for i in range(len(rs)):
            coords[i][0] = 0.0
            coords[i][1] = rs[i] * sin(self.thetas[i])
            coords[i][2] = rs[i] * cos(self.thetas[i])

        # create cubic interpolation functions
        interp_xs = interp1d(coords[-4:,2], coords[-4:,0], kind='cubic')
        interp_ys = interp1d(coords[-4:,2], coords[-4:,1], kind='cubic')

        # calculate x and y at given z value
        x = interp_xs(L_field)
        y = interp_ys(L_field)

        # replace last row of coords with interpolated values
        coords[-1] = np.array([x, y, L_field])
        return Streamline(xyz_coords=coords)

    def plot(self, Streamline, show_streamline=True, show_shock=True, 
        show_cone=True, show_plot=False, save_SVG=True, 
        file_name='conical_field'):
        # create plot of streamline
        fig = plt.figure(figsize=(16, 9))
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.size": 20
        })
        ax = plt.axes()
        max_z = abs(np.amax(Streamline.zs))
        axis_coords = np.array([[0, 0],
                                [max_z, 0]])
        ax.plot(axis_coords[:,0], axis_coords[:,1], 'k-.', 
            label='Axis of Symmetry')

        if show_streamline == True:
            ax.plot(Streamline.zs, Streamline.ys, 'b-', label='Streamline')

        if show_shock == True:
            shock_coords = np.array([[0, 0],
                                     [max_z, -max_z * tan(self.beta)]])
            ax.plot(shock_coords[:,0], shock_coords[:,1], 'r-', 
                label='Shockwave')
    
        if show_cone == True:
            cone_coords = np.array([[0, 0],
                                    [max_z, -max_z * tan(self.thetac)]])
            ax.plot(cone_coords[:,0], cone_coords[:,1], 'k-', label='Cone')

        ax.set_xlabel('$z$')
        ax.set_ylabel('$y$')
        plt.axis('equal')
        plt.grid()
        plt.legend()
        if show_plot == True:
            plt.show()
        if save_SVG == True:
            fig.savefig(file_name + '.svg', bbox_inches='tight')

    def cone_surface(self, L_field, n_r=100, n_phi=100):
        # generate surface grid for 3D cone
        rs = np.linspace(0, L_field*tan(self.thetac), n_r)
        phis = np.linspace(0, 2*pi, n_phi)
        cone_surf = np.nan * np.ones((n_r, n_phi, 3))
        for i in range(n_r):
            for j in range(n_phi):
                x = rs[i]*np.cos(phis[j])
                y = rs[i]*np.sin(phis[j])
                a = tan(self.thetac)

                cone_surf[i][j][0] = x
                cone_surf[i][j][1] = y
                cone_surf[i][j][2] = np.sqrt((x*x + y*y)/(a*a))
        return cone_surf

    def shock_surface(self, L_field, n_r=100, n_phi=100):
        # generate surface grid for 3D shock cone
        rs = np.linspace(0, L_field*tan(self.beta), n_r)
        phis = np.linspace(0, 2*pi, n_phi)
        shock_surf = np.nan * np.ones((n_r, n_phi, 3))
        for i in range(n_r):
            for j in range(n_phi):
                x = rs[i]*np.cos(phis[j])
                y = rs[i]*np.sin(phis[j])
                a = tan(self.beta)

                shock_surf[i][j][0] = x
                shock_surf[i][j][1] = y
                shock_surf[i][j][2] = np.sqrt((x*x + y*y)/(a*a))
        return shock_surf

def conical_M0_beta(config):
    # unpack dictionary
    M0 = config['mach_no']
    beta = config['shock_angle']
    gamma = config.get('rat_spec_heats', 1.4)
    dtheta = config.get('theta_step', 0.01*pi/180)
    max_steps = config.get('max_steps', 10000)
    interp_sing = config.get('interp_sing', True)
    print_freq = config.get('print_freq', 10)
    verbosity = config.get('verbosity', 1)

    # ODE initial conditions
    delta = theta_oblique(beta, M0, gamma)
    M1 = M2_oblique(beta, delta, M0, gamma)
    u1 = M1 * cos(beta-delta)
    v1 = -M1 * sin(beta-delta)

    # integration settings
    r = ode(taylor_maccoll_mach).set_integrator('DOP853', nsteps=max_steps)
    r.set_initial_value([u1, v1], beta)
    r.set_f_params(gamma)
    dt = dtheta

    # check singularity value before integrating
    if r.y[1] >= 0:
        text = 'v is >=0 before integration. \n'
        text += f'beta = {beta * 180/pi:.4} deg \n'
        text += f'[u, v] = {[u1, v1]}'
        raise ValueError(text)

    # intialise solution lists
    thetas = [beta]
    us = [u1]
    vs = [v1]

    # begin integration
    if verbosity == 1:
        print('Solving conical flow field...')
        width = 12
        print('-'*3*width)
        print(f"{'Step':^{width}}{'Theta':^{width}}{'Angular':^{width}}")
        print(f"{'Number':^{width}}{'(deg)':^{width}}{'Mach No.':^{width}}")
        print('-'*3*width)
    i = 0
    while r.successful() and r.y[1] < 0:
        r.integrate(r.t - dt)
        if verbosity == 1 and i % print_freq == 0:
            print(f"{i:^{width}}{r.t*180/pi:^{width}.4}{r.y[1]:^{width}.4}")
        thetas.append(r.t)
        us.append(r.y[0])
        vs.append(r.y[1])
        i += 1
    
    # use interpolation to find singularity point
    if interp_sing == True:
        # create cubic interpolation functions
        interp_thetas = interp1d(vs[-4:], thetas[-4:], kind='cubic')
        interp_us = interp1d(thetas[-4:], us[-4:], kind='cubic')

        # locate the theta value at the singularity
        theta_sing = float(interp_thetas(0))

        # interpolate u at theta_sing
        u_sing = float(interp_us(theta_sing))

        # replace last value of thetas, us and vs with interpolated values
        thetas[-1] = theta_sing
        us[-1] = u_sing
        vs[-1] = 0.0
    else:
        # remove singularity from solution
        thetas.pop(-1)
        us.pop(-1)
        vs.pop(-1)
        
    # print integration termination statement
    if not r.successful():
        raise AssertionError('Integration failed.')
    else:
        if verbosity == 1:
            # print reason for integration termination
            print('-'*3*width)
            print('Integration terminated successfully.\n')
        
            if interp_sing:
                print('Solution at interpolated singularity:')
            else:
                print('Solution at final step:')
            print(f'theta={thetas[-1] * 180/pi:.4} deg, u={us[-1]:.4}, ' + \
                f'v={vs[-1]:.4}\n')

    # return conical flow field object
    field = ConicalField(thetas, us, vs, M0, beta, thetas[-1], gamma)
    return field

def conical_M0_thetac(config):
    # unpack dictionary
    M0 = config['mach_no']
    thetac = config['cone_angle']
    gamma = config.get('rat_spec_heats', 1.4)
    beta_guess = config.get('beta_guess', 20*pi/180)
    tol = config.get('tol', 1.0E-6)
    dtheta = config.get('theta_step', 0.01*pi/180)
    max_steps = config.get('max_steps', 10000)
    interp_sing = config.get('interp_sing', True)
    print_freq = config.get('print_freq', 10)
    verbosity = config.get('verbosity', 1)

    config_new = config.copy()
    config_new['verbosity'] = 0
    
    def res(beta):
        # thetac residual function
        config_new['shock_angle'] = beta
        field = conical_M0_beta(config_new)
        res = thetac - field.thetac

        # print solver progress
        if verbosity == 1:
            global it
            beta0 = beta[0] * 180/pi
            print(f'{it:^{width}}{beta0:^{width}.4}{res:^{width}.2e}')
            it += 1
        return res
        
    # use root finder to iterate residual function
    if verbosity == 1:
        print('Using root finder to calculate shock angle...')
        width = 12
        print('-'*3*width)
        print(f"{'Iteration':^{width}}{'Shock':^{width}}{'Angle':^{width}}")
        print(f"{'Number':^{width}}{'Angle (deg)':^{width}}{'Residual':^{width}}")
        print('-'*3*width)
    global it
    it = 0
    sol = root(res, beta_guess, method='hybr', tol=tol)
    if sol.success == False:
        raise AssertionError('Root finder failed to converge.')
    else:
        if verbosity == 1:
            print('-'*3*width)
            print('Root finder successfully converged.\n')
    
    # return conical field object
    beta = sol.x[0]
    config_new = config.copy()
    config['shock_angle'] = beta
    field = conical_M0_beta(config)
    return field

def eval_flow_data(field, mesh, free_stream):
    n_i = len(mesh)
    n_j = len(mesh[0])
    
    labels = ['x', 'y', 'z', 'theta', 'delta', 'mach_no', 'press', 'temp']
    flow_data = {}
    for label in labels:
        flow_data[label] = np.nan * np.ones((n_i, n_j))

    for i in range(n_i):
        for j in range(n_j):
            x_ij = mesh[i][j][0]
            y_ij = mesh[i][j][1]
            z_ij = mesh[i][j][2]
            theta_ij = atan(sqrt(x_ij**2 + y_ij**2)/z_ij)

            flow_data['x'][i][j] = x_ij
            flow_data['y'][i][j] = y_ij
            flow_data['z'][i][j] = z_ij
            flow_data['theta'][i][j] = theta_ij
            flow_data['delta'][i][j] = atan(field.v(theta_ij)/field.u(theta_ij))
            flow_data['mach_no'][i][j] = field.M(theta_ij)
            flow_data['press'][i][j] = field.p(theta_ij, free_stream['press'])
            flow_data['temp'][i][j] = field.T(theta_ij, free_stream['temp'])

    return flow_data

def flow_data_to_vtk(flow_data, file_name='flow_data'):
    n_i = len(flow_data['x'])
    n_j = len(flow_data['x'][0])

    s_grid = vtkStructuredGrid()
    s_grid.SetDimensions([n_i, n_j, 1])
    points = vtkPoints()
    points.Allocate(n_i*n_j*1)

    x = vtkDoubleArray()
    y = vtkDoubleArray()
    z = vtkDoubleArray()
    theta = vtkDoubleArray()
    delta = vtkDoubleArray()
    mach_no = vtkDoubleArray()
    press = vtkDoubleArray()
    temp = vtkDoubleArray()
    
    x.SetName('x')
    y.SetName('y')
    z.SetName('z')
    theta.SetName('Theta (rad)')
    delta.SetName('Flow Angle (rad)')
    mach_no.SetName('Mach Number')
    press.SetName('Pressure (Pa)')
    temp.SetName('Temperature (K)')

    for j in range(n_j):
        j_offset = j*n_i
        for i in range(n_i):
            offset = i + j_offset
            point_ij = [flow_data['x'][i][j], flow_data['y'][i][j], 
                        flow_data['z'][i][j]]
            points.InsertPoint(offset, point_ij)
            x.InsertTuple(offset, [flow_data['x'][i][j]])
            y.InsertTuple(offset, [flow_data['y'][i][j]])
            z.InsertTuple(offset, [flow_data['z'][i][j]])
            theta.InsertTuple(offset, [flow_data['theta'][i][j]])
            delta.InsertTuple(offset, [flow_data['delta'][i][j]])
            mach_no.InsertTuple(offset, [flow_data['mach_no'][i][j]])
            press.InsertTuple(offset, [flow_data['press'][i][j]])
            temp.InsertTuple(offset, [flow_data['temp'][i][j]])
    
    s_grid.SetPoints(points)
    s_grid.GetPointData().AddArray(x)
    s_grid.GetPointData().AddArray(y)
    s_grid.GetPointData().AddArray(z)
    s_grid.GetPointData().AddArray(theta)
    s_grid.GetPointData().AddArray(delta)
    s_grid.GetPointData().AddArray(mach_no)
    s_grid.GetPointData().AddArray(press)
    s_grid.GetPointData().AddArray(temp)

    writer = vtkXMLStructuredGridWriter()
    writer.SetInputData(s_grid)
    writer.SetFileName(file_name + '.vtu')
    writer.SetDataModeToAscii()
    writer.Update()

def avg_flow_data(flow_data):
    avg_props = flow_data.copy()
    for key in avg_props:
        avg_props[key] = np.average(avg_props[key])
    return avg_props
