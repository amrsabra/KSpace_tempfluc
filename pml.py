# pml.py

import numpy as np
import constants

# Fallback, for if C-Compilation doesn't work out, use NumPy
try:
    from pml_python_c import update_pml_c
    C_PML = True
except ImportError:
    C_PML = False
    print("C-Compilation didn't happen. Using NumPy fallback")

class PML:
    def __init__(self, N, dx, c0=constants.C0, depth=constants.PML_DEPTH, absorption=constants.PML_ABSORPTION):
        self.N = N
        self.dx = dx
        self.c0 = c0
        self.depth = depth
        self.A = absorption  # Absorption strength in nepers
        self._init_absorption_profiles()
        self.dt = None 
        
    def _init_absorption_profiles(self):
        self.alpha_x = np.zeros(self.N) # Initialise 1D array of zeros instead of 3D array to save memory.
        self.alpha_y = np.zeros(self.N)
        self.alpha_z = np.zeros(self.N)
        
        # For each dimension, create absorption profiles with 1D arrays.
        for dim, alpha_array in [('x', self.alpha_x), ('y', self.alpha_y), ('z', self.alpha_z)]:
            # Left boundary
            for i in range(self.depth):
                # Distance from outer edge.
                x_rel = (self.depth - 1 - i) / self.depth
                # EQN 14: quartic taper
                alpha_array[i] = self.A * (self.c0 / self.dx) * (x_rel ** 4)

            # Right boundary  
            for i in range(self.depth):
                # Distance from outer edge.
                x_rel = i / self.depth
                # EQN 14: quartic taper
                alpha_array[self.N - self.depth + i] = self.A * (self.c0 / self.dx) * (x_rel ** 4)

    def set_dt(self, dt):
        # Precompute exponential terms
        self.dt = dt
        self.exp_half_x = np.exp(self.alpha_x * dt / 2)
        self.exp_half_y = np.exp(self.alpha_y * dt / 2)
        self.exp_half_z = np.exp(self.alpha_z * dt / 2)

        self.exp_neg_half_x = 1.0 / self.exp_half_x
        self.exp_neg_half_y = 1.0 / self.exp_half_y
        self.exp_neg_half_z = 1.0 / self.exp_half_z

        
    '''
    update_velocity and update_pressure implement time-stepping for a damped wave equation.
    the exp ensures update stays stable with high absorption, as seen in EQN 13
    '''
    def update_velocity_component(self, u_prev, rhs, dt, direction):
        # Update component using C or NumPy fallback
        if self.dt != dt:
            self.set_dt(dt)
        
        if C_PML:
            if direction == 'x':
                ex, en = self.exp_half_x, self.exp_neg_half_x
            elif direction == 'y':
                ex, en = self.exp_half_y, self.exp_neg_half_y
            elif direction == 'z':
                ex, en = self.exp_half_z, self.exp_neg_half_z
            else:
                raise ValueError("Direction must be 'x', 'y', or 'z'")
            
            update_pml_c(u_prev, rhs, ex, en, dt, direction)
            return u_prev
        else:
            # NumPy Fallback
            if direction == 'x':
                e_h, e_n = self.exp_half_x[:, None, None], self.exp_neg_half_x[:, None, None]
            elif direction == 'y':
                e_h, e_n = self.exp_half_y[None, :, None], self.exp_neg_half_y[None, :, None]
            else:
                e_h, e_n = self.exp_half_z[None, None, :], self.exp_neg_half_z[None, None, :]
        
            u_prev *= e_n
            u_prev += (dt * rhs)
            u_prev /= e_h
            return u_prev
    
    def update_pressure_component(self, p_prev, rhs, dt, direction):
        """Applies the same update logic to pressure components."""
        return self.update_velocity_component(p_prev, rhs, dt, direction)