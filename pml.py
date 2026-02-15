# pml.py

import numpy as np
import constants

class PML:
    def __init__(self, N, dx, c0=constants.C0, depth=constants.PML_DEPTH, absorption=constants.PML_ABSORPTION):
        self.N = N
        self.dx = dx
        self.c0 = c0
        self.depth = depth
        self.A = absorption  # Absorption strength in nepers
        
        self._init_absorption_profiles()

        self.exp_neg_alpha_x_dt_half = None
        self.exp_neg_alpha_y_dt_half = None
        self.exp_neg_alpha_z_dt_half = None

        
    def _init_absorption_profiles(self):
        self.alpha_x = np.zeros(self.N) # Initialise array of zeros.
        self.alpha_y = np.zeros(self.N)
        self.alpha_z = np.zeros(self.N)
        
        # For each dimension, create absorption profiles with 1D arrays.
        for dim, alpha_array in [('x', self.alpha_x), 
                                  ('y', self.alpha_y), 
                                  ('z', self.alpha_z)]:
            
            # Left boundary
            for i in range(self.depth):
                # Distance from outer edge.
                x_rel = (self.depth - 1 - i) / self.depth
                
                # EQN 14: quartic taper
                alpha_val = self.A * (self.c0 / self.dx) * (x_rel ** 4)
                alpha_array[i] = alpha_val
            
            # Right boundary  
            for i in range(self.depth):
                # Distance from outer edge.
                x_rel = i / self.depth
                
                # EQN 14: quartic taper
                alpha_val = self.A * (self.c0 / self.dx) * (x_rel ** 4)
                alpha_array[self.N - self.depth + i] = alpha_val
        
        # 3D absorption arrays 
        self.alpha_x_3d = self.alpha_x[:, None, None]
        self.alpha_y_3d = self.alpha_y[None, :, None]
        self.alpha_z_3d = self.alpha_z[None, None, :]
        
        # Precompute exponential terms for efficiency, used in EQN 13
        self.exp_alpha_x_dt_half = None
        self.exp_alpha_y_dt_half = None
        self.exp_alpha_z_dt_half = None

    '''  
    # Update equations have deltat, so we are setting up the component.
    # since deltat doesnt change, we precalculate it to save time. 
    '''
    def set_dt(self, dt):
        self.dt = dt

        self.exp_alpha_x_dt_half = np.exp(self.alpha_x_3d * dt / 2)
        self.exp_alpha_y_dt_half = np.exp(self.alpha_y_3d * dt / 2)
        self.exp_alpha_z_dt_half = np.exp(self.alpha_z_3d * dt / 2)

        self.exp_neg_alpha_x_dt_half = 1.0 / self.exp_alpha_x_dt_half
        self.exp_neg_alpha_y_dt_half = 1.0 / self.exp_alpha_y_dt_half
        self.exp_neg_alpha_z_dt_half = 1.0 / self.exp_alpha_z_dt_half

        
    '''
    update_velocity and update_pressure implement time-stepping for a damped wave equation.
    the exp ensures update stays stable with high absorption, as seen in EQN 13
    '''
    def update_velocity_component(self, u_prev, rhs, dt, direction):
        if self.exp_alpha_x_dt_half is None:
            self.set_dt(dt)
        
        if direction == 'x':
            exp_half = self.exp_alpha_x_dt_half
            exp_neg_half = self.exp_neg_alpha_x_dt_half
        elif direction == 'y':
            exp_half = self.exp_alpha_y_dt_half
            exp_neg_half = self.exp_neg_alpha_y_dt_half
        elif direction == 'z':
            exp_half = self.exp_alpha_z_dt_half
            exp_neg_half = self.exp_neg_alpha_z_dt_half
        else:
            raise ValueError("Direction must be 'x', 'y', or 'z'")

        # EQN 13: exponential time-stepping (IN-PLACE)
        # Formula: u_new = (exp_neg_half * u_prev + dt * rhs) / exp_half
        
        u_prev *= exp_neg_half
        u_prev += (dt * rhs)
        u_prev /= exp_half
        
        return u_prev
    
    def update_pressure_component(self, p_prev, rhs, dt, direction):
        if self.exp_alpha_x_dt_half is None:
            self.set_dt(dt)
        
        if direction == 'x':
            exp_half = self.exp_alpha_x_dt_half
            exp_neg_half = self.exp_neg_alpha_x_dt_half
        elif direction == 'y':
            exp_half = self.exp_alpha_y_dt_half
            exp_neg_half = self.exp_neg_alpha_y_dt_half
        elif direction == 'z':
            exp_half = self.exp_alpha_z_dt_half
            exp_neg_half = self.exp_neg_alpha_z_dt_half
        else:
            raise ValueError("Direction must be 'x', 'y', or 'z'")
        
        # EQN 13: exponential time-stepping (IN-PLACE)
        
        p_prev *= exp_neg_half
        p_prev += (dt * rhs)
        p_prev /= exp_half
        
        return p_prev