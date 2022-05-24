import json
import os
import csv
from eilmer.spline import CubicSpline

# read inflow parameters from json file
f = open('inflow.json')
inflow = json.load(f)
f.close()

config.axisymmetric = True
L_ext = 0.25

init_gas_model('ideal-air-gas-model.lua')
gas1 = GasState(config.gmodel)
gas1.p = inflow['press']
gas1.T = inflow['temp']
gas1.update_thermo_from_pT()
gas1.update_sound_speed()
M1 = inflow['mach_no']
V1 = M1 * gas1.a

config.max_step_relax = 40


xs = []; ys = []
current_dir = os.getcwd()
sol_no = int(current_dir.split('_')[-1])
with open(f'contour_{sol_no}.csv', 'r') as df:
    pt_data = csv.reader(df, delimiter=' ')
    next(df)
    for row in pt_data:
        xs.append(float(row[0]))
        ys.append(float(row[1]))

x_thrt = xs[-1]
y_thrt = ys[-1]

busemann_contour = CubicSpline(xs, ys)
def upper_y(x):
    return busemann_contour(x) if x < x_thrt else y_thrt
def lower_y(x): return 0.0
def lower_bc(x): return 0
def upper_bc(x): return 0

config.max_x = x_thrt + L_ext
config.dx = config.max_x/1000

st1 = StreamTube(gas=gas1, velx=V1, vely=0.0,
                 y0=lower_y, y1=upper_y,
                 bc0=lower_bc, bc1=upper_bc,
                 ncells=75)
