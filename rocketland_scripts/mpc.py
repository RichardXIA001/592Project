import numpy as np
import sys
from casadi import *

# Add do_mpc to path. This is not necessary if it was installed via pip
import os
rel_do_mpc_path = os.path.join('..','..','..')
sys.path.append(rel_do_mpc_path)
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

# Import do_mpc package:
import do_mpc

import matplotlib.pyplot as plt

import dataio, utils, training, loss_functions

import configargparse
from argparse import ArgumentParser
import json
## model
model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

## states and control inputs

# Load the dataset
model_name = 'rocket1'
opt_path = os.path.join('./logs',model_name, 'commandline_args.txt')
parser = ArgumentParser()
opt = parser.parse_args()
with open(opt_path, 'r') as f:
    opt.__dict__ = json.load(f)
dataset = dataio.ReachabilityParameterConditionedSimpleRocketLandingSource(numpoints=65000, pretrain=opt.pretrain, tMin=opt.tMin,
                                                                           tMax=opt.tMax, counter_start=opt.counter_start, counter_end=opt.counter_end,
                                                                           pretrain_iters=opt.pretrain_iters, num_src_samples=opt.num_src_samples,
                                                                         diffModel=opt.diffModel, num_target_samples=opt.num_target_samples, lxType=opt.lxType)
# states variable
Y = model.set_variable(var_type='_x', var_name='Y', shape=(1,1))
dY = model.set_variable(var_type='_x', var_name='dY', shape=(1,1))
Z = model.set_variable(var_type='_x', var_name='Z', shape=(1,1))
dZ = model.set_variable(var_type='_x', var_name='dZ', shape=(1,1))
Theta = model.set_variable(var_type='_x', var_name='Theta', shape=(1,1))
dTheta = model.set_variable(var_type='_x', var_name='dTheta', shape=(1,1))

# input variable
U1 = model.set_variable(var_type='_u', var_name='U1')
U2 = model.set_variable(var_type='_u', var_name='U2')

model.set_rhs('Y', dY)
model.set_rhs('dY', U1*np.cos(Theta) - U2*np.sin(Theta))
model.set_rhs('Z', dZ)
model.set_rhs('dZ', U1*np.sin(Theta) + U2*np.cos(Theta) - dataset.dynSys['g'])
model.set_rhs('Theta', dTheta)
model.set_rhs('dTheta', dataset.dynSys['alpha'] * U1)
model.setup()

# ********** Setting up MPC controller **********
mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 20,
    'n_robust': 1,
    'open_loop': 0,
    't_step': 0.005,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 2,
    'collocation_ni': 2,
    'store_full_solution': True,
    # Use MA27 linear solver in ipopt for faster calculations:
    #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
}

mpc.set_param(**setup_mpc)

mpc.scaling['_x', 'Y'] = dataset.alpha['y']
mpc.scaling['_x', 'dY'] = dataset.alpha['y_dot']
mpc.scaling['_x', 'Z'] = dataset.alpha['z']
mpc.scaling['_x', 'dZ'] = dataset.alpha['z_dot']
mpc.scaling['_x', 'Theta'] = dataset.alpha['th']
mpc.scaling['_x', 'dTheta'] = dataset.alpha['th_dot']


# TODO *************Objective function*************
beta = 0.0
_x = model.x

x_unnormalized = np.array([_x['Y'], _x['Z'], _x['Theta'], _x['dY'], _x['dZ'], _x['dTheta'], beta])
print("x_unnormalized", x_unnormalized)

def compute_lx(_x):
    dist_y = sqrt((_x['Y'] - beta)**2) - dataset.dynSys['max_y']
    
    dist_z = _x['Z'] - dataset.dynSys['L'] - dataset.dynSys['max_z']
    
    lx = fmax(dist_y, dist_z)
    
    condition = lx >= 0
    positive_case = lx / 150.0
    negative_case = lx / 10.0
    
    lx = if_else(condition, positive_case, negative_case)

    return lx
def residual_loss(_x, target, time_horizon=10):
    # loss = []
    y_resid = _x['Y'] - target[0]
    z_resid = _x['Z'] - target[1]
    
    loss = fabs(y_resid) + fabs(z_resid)
    # print("_x['Y']", _x['Y'])
    return y_resid

mterm = compute_lx(_x)
lterm = residual_loss(_x, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])) 

mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(U1=0.01, U2=0.01)


# Bounds
CONTROL_BOUNDS = dataset.dynSys['thrustMax'] / sqrt(2)
mpc.bounds['lower','_u', 'U1'] = -CONTROL_BOUNDS
mpc.bounds['upper','_u', 'U1'] = CONTROL_BOUNDS

mpc.bounds['lower','_u', 'U2'] = -CONTROL_BOUNDS
mpc.bounds['upper','_u', 'U2'] = CONTROL_BOUNDS

mpc.bounds['lower', '_x', 'Y'] = -dataset.alpha['y']
mpc.bounds['upper', '_x', 'Y'] = dataset.alpha['y']

mpc.bounds['lower', '_x', 'dY'] = -dataset.alpha['y_dot']
mpc.bounds['upper', '_x', 'dY'] = dataset.alpha['y_dot']

mpc.bounds['lower', '_x', 'Z'] = -dataset.alpha['z']
mpc.bounds['upper', '_x', 'Z'] = dataset.alpha['z']

mpc.bounds['lower', '_x', 'dZ'] = -dataset.alpha['z_dot']
mpc.bounds['upper', '_x', 'dZ'] = dataset.alpha['z_dot']

mpc.bounds['lower', '_x', 'Theta'] = -dataset.alpha['th']
mpc.bounds['upper', '_x', 'Theta'] = dataset.alpha['th']

mpc.bounds['lower', '_x', 'dTheta'] = -dataset.alpha['th_dot']
mpc.bounds['upper', '_x', 'dTheta'] = dataset.alpha['th_dot']
mpc.setup()
# all states can be directly measured (state-feedback)
estimator = do_mpc.estimator.StateFeedback(model)

simulator = do_mpc.simulator.Simulator(model)

params_simulator = {
    'integration_tool': 'cvodes',
    'abstol': 1e-10,
    'reltol': 1e-10,
    't_step': 0.005
}

simulator.set_param(**params_simulator)

simulator.setup()


# Closed-loop simulation

x0 = np.array([30.0, 0.0, 60.0, -10.0, 0.0, 0.0])

mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()

for k in range(500):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
from matplotlib import rcParams
rcParams['axes.grid'] = True
rcParams['font.size'] = 18

fig, ax = plt.subplots(5, sharex=True, figsize=(16,24))
mpc_graphics.add_line(var_type='_x', var_name='Y', axis=ax[0])
mpc_graphics.add_line(var_type='_x', var_name='Z', axis=ax[1])

mpc_graphics.add_line(var_type='_x', var_name='dY', axis=ax[2])
mpc_graphics.add_line(var_type='_x', var_name='dZ', axis=ax[3])

mpc_graphics.add_line(var_type='_x', var_name='Theta', axis=ax[4])
# mpc_graphics.add_line(var_type='_x', var_name='dTheta', axis=ax[5])

# mpc_graphics.add_line(var_type='_u', var_name='U1', axis=ax[6])
# mpc_graphics.add_line(var_type='_u', var_name='U2', axis=ax[7])

ax[0].set_ylabel('Y')
ax[1].set_ylabel('Z')
ax[2].set_ylabel('dY')
ax[3].set_ylabel('dZ')
ax[4].set_ylabel('Theta')
# ax[5].set_ylabel('dTheta')
# ax[6].set_ylabel('U1')
# ax[7].set_ylabel('U2')


for line_i in mpc_graphics.pred_lines.full:
    line_i.set_linewidth(2)
# for line_i in np.sum(mpc_graphics.pred_lines['_x', :, :,0]):
#     line_i.set_linewidth(5)
# for line_i in np.sum(mpc_graphics.pred_lines['_u', :, :,0]):
#     line_i.set_linewidth(5)


from matplotlib.animation import FuncAnimation, ImageMagickWriter, FFMpegWriter

def update(t_ind):
    print('Writing frame: {}.'.format(t_ind), end='\r')
    mpc_graphics.plot_results(t_ind=t_ind)
    mpc_graphics.plot_predictions(t_ind=t_ind)
    mpc_graphics.reset_axes()
    lines = mpc_graphics.result_lines.full
    return lines

n_steps = mpc.data['_time'].shape[0]


anim = FuncAnimation(fig, update, frames=n_steps, blit=True)

video_writer = FFMpegWriter(fps=5)
anim.save('rocket_mpc_500.mp4', writer=video_writer)
# gif_writer = ImageMagickWriter(fps=5)

# anim.save('rocket_mpc.gif', writer=gif_writer)