# incident_wave.py
# This file models the vertical incident wave moving in direction +z. 

import numpy as np
import constants

class IncidentWave:
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z
        
    def plane_wave(self, t, fm=constants.DEFAULT_FM, sigma=constants.DEFAULT_TAU, mu=constants.DEFAULT_DELAY):
        
        tau = t - self.Z / constants.C0 # Delay due to wave propagation, inside EQN 1
        
        # Gaussian-modulated wave, search "modulated gaussian plane wave"
        exponential = np.exp(-((tau - mu) / (2 * sigma))**2)
        carrier = np.cos(2 * np.pi * fm * (tau - mu))
        
        # Incident pressure
        p_i = exponential * carrier # EQN 1
        
        # Incident particle velocity (plane wave relation)
        u_i_z = p_i / (constants.RHO0 * constants.C0) # EQN 2
        
        return p_i, u_i_z