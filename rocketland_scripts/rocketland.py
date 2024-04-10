# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, training, loss_functions, modules, diff_operators


import torch
import numpy as np
import scipy
from scipy import linalg
import math
from torch.utils.data import DataLoader
import configargparse
import scipy.io as spio

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
import math
from torch.utils.data import DataLoader
import configargparse
from argparse import ArgumentParser

class LunarLanding(dataio.ReachabilityParameterConditionedSimpleRocketLandingSource):
    def __init__(self, numpoints, pretrain=False, tMin=0, tMax=1, counter_start=0, counter_end=100000,\
        pretrain_iters=10000, num_src_samples=10000, num_target_samples=10000, diffModel=False, lxType='unit_normalized_max'):
        super().__init__(numpoints, pretrain, tMin, tMax, counter_start, counter_end, pretrain_iters,\
            num_src_samples, num_target_samples, diffModel, lxType)
        
        self.low = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                -1.5,
                -1.5,
                # velocity bounds is 5x rated speed
                -5.0,
                -5.0,
                -math.pi,
                -5.0,
                # -0.0,
                # -0.0,
            ]
        ).astype(np.float32)
        self.high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                1.5,
                1.5,
                # velocity bounds is 5x rated speed
                5.0,
                5.0,
                math.pi,
                5.0,
                # 1.0,
                # 1.0,
            ]
        ).astype(np.float32)

        self.H = 400 / 30.0
        self.W = 600 / 30.0
        self.FPS = 50
        self.LEG_DOWN = 18 / 30.0
        
        self.CHUNKS = 11
        
        height = np.random.uniform(0, self.H / 2, size=(self.CHUNKS + 1,))
        chunk_x = [self.W / (self.CHUNKS - 1) * i for i in range(self.CHUNKS)]
        self.helipad_x1 = chunk_x[self.CHUNKS // 2 - 1] - self.W / 2
        self.helipad_x2 = chunk_x[self.CHUNKS // 2 + 1] - self.W / 2
        self.helipad_y = self.H / 4
        height[self.CHUNKS // 2 - 2] = self.helipad_y
        height[self.CHUNKS // 2 - 1] = self.helipad_y
        height[self.CHUNKS // 2 + 0] = self.helipad_y
        height[self.CHUNKS // 2 + 1] = self.helipad_y
        height[self.CHUNKS // 2 + 2] = self.helipad_y
        self.dynSys['thrustMax'] = 13.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.num_states = 6
    
        self.norm_to = 0.02
        self.mean = 0.0
        self.var = 1.0
    
    def dynamics(unnormalized_states, controls, disturbance=None):
        '''
        unnormalized_states: [batch_size, num_dims] (x, y, vx, vy, theta, vtheta)
        
        controls: [batch_size, num_controls] (main engine, side engine)
        '''
        # predefined mass, gravity, and moment of inertia
        M = 4.85
        G = 10.0
        J = None
        
        theta = unnormalized_states[:, 4]
        
        u1 = controls[:, 0]
        u2 = controls[:, 1]
        
        ddx = (u1 * np.sin(theta) + u2 * np.cos(theta)) / M
        # ddy = (u1 * np.cos(theta) - u2 * np.sin(theta) - M * (G /6.666)) / M
        ddy = (u1 * np.cos(theta) - u2 * np.sin(theta)  ) / M
        dtheta = u2 * ( 14 / 30 ) / J
        
        ddx = -2.63292989 * ddx + (-0.02091248)
        
        ddy = 1.38131398 * ddy + (-0.83123823)
        
        
        return 0.5 * np.array([(100 / 97) * unnormalized_states[:, 2], (200/ 97)* unnormalized_states[:, 3], ddx, ddy, 5* unnormalized_states[:, 5], dtheta]).T
        
    def step_to_next_states(self, unnormalized_states, controls,disturbance=None, dt=0.02):
        '''
        unnormalized_states: [batch_size, num_dims] (x, y, vx, vy, theta, vtheta)
        
        controls: [batch_size, num_controls] (main engine, side engine)
        '''
        derivates = self.dynamics(unnormalized_states, controls, disturbance)
        next_states = unnormalized_states + derivates * dt
        # constrain the angle to be within [-pi, pi]
        next_states[:, 4] = (next_states[:, 4] + np.pi) % (2 * np.pi) - np.pi
        return next_states
    
    # normalize the states
    def normalize_states(self, unnormalized_states):
        # normalize the states
        # judge the type of the input states and adjust the normalization accordingly
        if type(unnormalized_states) == torch.Tensor:
            device = unnormalized_states.device
            normalized_states = (unnormalized_states - torch.tensor(self.low).to(device)) / (torch.tensor(self.high).to(device) - torch.tensor(self.low).to(device))
        elif type(unnormalized_states) == np.ndarray:
            normalized_states = (unnormalized_states - self.low) / (self.high - self.low)
        else:
            raise ValueError('Input states should be either torch.Tensor or np.ndarray')
        return normalized_states
    
    def unnormalize_states(self, normalized_states):
        # unnormalize the states
        # judge the type of the input states and adjust the normalization accordingly
        if type(normalized_states) == torch.Tensor:
            device = normalized_states.device
            unnormalized_states = normalized_states * (torch.tensor(self.high).to(device) - torch.tensor(self.low).to(device)) + torch.tensor(self.low).to(device)
        elif type(normalized_states) == np.ndarray:
            unnormalized_states = normalized_states * (self.high - self.low) + self.low
        else:
            raise ValueError('Input states should be either torch.Tensor or np.ndarray')
        return unnormalized_states
    
    def unnormalize_dVdX(self, normalized_dVdX):
        # unnormalize the states
        # judge the type of the input states and adjust the normalization accordingly
        if type(normalized_dVdX) == torch.Tensor:
            device = normalized_dVdX.device
            unnormalized_dVdX = normalized_dVdX / (torch.tensor(self.high).to(device) - torch.tensor(self.low).to(device))
        elif type(normalized_dVdX) == np.ndarray:
            unnormalized_dVdX = normalized_dVdX / (self.high - self.low)
        else:
            raise ValueError('Input states should be either torch.Tensor or np.ndarray')
        return unnormalized_dVdX
    
    def state_to_position(self, unnormalized_states):
        pos = unnormalized_states.clone()
        pos[:, 0] = unnormalized_states[:, 0] * (self.W / 2) + self.W / 2
        pos[:, 1] = unnormalized_states[:, 1] * (self.H / 2) + self.H / 4 + self.LEG_DOWN
        pos[:, 2] = unnormalized_states[:, 2] *  self.FPS / (self.W / 2)
        pos[:, 3] = unnormalized_states[:, 3] *  self.FPS / (self.H / 2)
        pos[:, 5] = unnormalized_states[:, 5] *  self.FPS / 20.0
        return pos
    
    def compute_lx(self, state_coords_normalized):
        state_coords_unnormalized = self.unnormalize_states(state_coords_normalized)
        
        pos = self.state_to_position(state_coords_unnormalized)
        
        dist_x = torch.abs(pos[:, 0])  - self.W / 2
        
        dist_y = pos[:, 1] - self.H / 4 - self.LEG_DOWN
        
        lx = torch.max(dist_x, dist_y)
        lx = torch.where((lx >=0), lx/self.W * 2, lx / (self.LEG_DOWN))
        
        return lx
        
    def sample_inside_target_set(self, num_samples):
        target_coords = torch.zeros(num_samples, self.num_states).uniform_(-1, 1).to(self.device)
        
        target_coords[:, 0] = target_coords[:, 0] * (self.helipad_x2 - self.helipad_x1) / 2 + (self.helipad_x2 + self.helipad_x1) / 2
        target_coords[:, 0] = (target_coords[:, 0] - self.W / 2) / (self.W / 2)
        
        target_coords[:, 1] = target_coords[:, 1] * 0.75 * (self.LEG_DOWN) + 1.75 * self.LEG_DOWN
        target_coords[:, 1] = (target_coords[:, 1] - self.H / 4 - self.LEG_DOWN) / (self.H / 2)
        
        return target_coords
    
    def compute_overall_ham(self, x, dudx, compute_xdot=False):
        return super().compute_overall_ham(x, dudx, compute_xdot)
    
    