# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, training, loss_functions, modules

import torch
import numpy as np
import scipy
import math
from torch.utils.data import DataLoader
import configargparse

checkpoint_toload=119000
# Load the model
model = modules.SingleBVPNet(in_features=5, out_features=1, type='sine', mode='mlp', final_layer_factor=1., hidden_features=512, num_hidden_layers=3)
model.cuda()
root_path = os.path.join('./deepreach_uncertain_parameter/air3D_scripts/logs/', '3Dp1D_u0')
root_path = os.path.join('./logs', 'air3D1')
ckpt_dir = os.path.join(root_path, 'checkpoints')
ckpt_path = os.path.join(ckpt_dir, 'model_epoch_%04d.pth' % checkpoint_toload)
checkpoint = torch.load(ckpt_path)
# checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
try:
  model_weights = checkpoint['model']
except:
  model_weights = checkpoint
model.load_state_dict(model_weights)
model.eval()

angle_alpha=1.2
epoch=checkpoint_toload
tMax=1.0
omega_a=5.5

times = [0.5*tMax]
times = [0.]

num_times = len(times)

# Theta slices to be plotted
thetas = [0.5*math.pi]
# thetas = [math.pi]
num_thetas = len(thetas)

# Create a figure
fig = plt.figure(figsize=(5, 5))

# Get the meshgrid in the (x, y) coordinate
sidelen = 200
mgrid_coords = dataio.get_mgrid(sidelen)

# Start plotting the results
time_coords = torch.ones(mgrid_coords.shape[0], 1) * times[0]
theta_coords = torch.ones(mgrid_coords.shape[0], 1) * thetas[0]
theta_coords = theta_coords / (angle_alpha * math.pi)
omega_a_coords = torch.ones(mgrid_coords.shape[0], 1) * omega_a  #plot for wa= selected value
omega_a_coords = (omega_a_coords - 4) / 1.5                      #scale back to (-1,+1)
coords = torch.cat((time_coords, mgrid_coords, theta_coords, omega_a_coords), dim=1) 
model_in = {'coords': coords.cuda()}
model_out = model(model_in)['model_out']

# Detatch model ouput and reshape
model_out = model_out.detach().cpu().numpy()
model_out = model_out.reshape((sidelen, sidelen))

# Unnormalize the value function
norm_to = 0.02
mean = 0.25
var = 0.5
model_out = (model_out*var/norm_to) + mean 

# Plot the zero level sets

# include 0. in levels

# pfs
model_out = utils.estimate_failure_probability(model_out)
min_value = np.min(model_out)
max_value = np.max(model_out)
levels = np.linspace(min_value, max_value, 5)
print("min, max", min_value, max_value)
# levels = np.sort(np.concatenate((levels, [0.])))

ax = fig.add_subplot(num_times, num_thetas, (0+1) + 0*num_thetas)
ax.set_title('t = %0.2f, theta = %0.2f' % (times[0], thetas[0]))
s = ax.contour(mgrid_coords[:, 0].reshape(sidelen, sidelen), mgrid_coords[:, 1].reshape(sidelen, sidelen), model_out, levels=levels, colors='k')
ax.clabel(s, inline=True, fontsize=8)
ax.pcolor(mgrid_coords[:, 0].reshape(sidelen, sidelen), mgrid_coords[:, 1].reshape(sidelen, sidelen), model_out, cmap='bwr', alpha=0.5)

model_out = (model_out <= 0.001)*1.

# Plot the actual data
# ax = fig.add_subplot(num_times, num_thetas, (0+1) + 0*num_thetas)
# ax.set_title('t = %0.2f, theta = %0.2f' % (times[0], thetas[0]))
# s = ax.imshow(model_out.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
# fig.colorbar(s) 

omega_index=omega_a*10
th_index=thetas[0]*100
t_index=times[0]*10

path = os.path.join(root_path, 'Pfs_plot_omega_%.2d_th_%.2d_t_%.2d.png' % (omega_index,th_index,t_index))
fig.savefig(path)
# print
print('Saved figure at: %s' % path)
scipy.io.savemat(os.path.join(root_path, 'Pfs_omega_%.2d_th_%.3d_t_%.2d.mat' % (omega_index,th_index,t_index)), {'model_out': model_out})