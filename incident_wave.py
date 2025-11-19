# incident_wave.py
# This file models the vertical incident wave moving in direction +z. 

import numpy as np
import constants
from constants import DEFAULT_FM, DEFAULT_TAU, DEFAULT_DELAY

class IncidentWave:
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z
        
    def plane_wave(self, t, fm=DEFAULT_FM, sigma=DEFAULT_TAU, mu=DEFAULT_DELAY):
        
        tau = t - self.Z / constants.C0 # Delay due to wave propagation
        
        # Gaussian-modulated wave
        exponential = np.exp(-((tau - mu) / (2 * sigma))**2)
        carrier = np.cos(2 * np.pi * fm * (tau - mu))
        
        # Incident pressure
        p_i = exponential * carrier # EQN 1
        
        # Incident particle velocity (plane wave relation)
        u_i_z = p_i / (constants.RHO0 * constants.C0) # EQN 2
        
        return p_i, u_i_z