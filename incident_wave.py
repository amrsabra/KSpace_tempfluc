# incident_wave.py
# This file models the vertical incident wave moving in direction +z. 

import numpy as np
import constants

class IncidentWave:
    def __init__(self, X, Y, Z):
        # We only need variation along z
        # Assume Z is a regular grid with shape (N, N, N)
        # use the first column along x and y
        z_axis = Z[0, 0, :]

        self.z_axis = z_axis
        self.z_over_c0 = z_axis / constants.C0
        self.div_rho0_c0 = 1.0 / (constants.RHO0 * constants.C0)

    def plane_wave(self, t, fm=constants.DEFAULT_FM, sigma=constants.DEFAULT_TAU, mu=constants.DEFAULT_DELAY):
        tau_1d = t - self.z_over_c0

        exponential_1d = np.exp(-((tau_1d - mu) / (2 * sigma))**2)
        carrier_1d = np.cos(2 * np.pi * fm * (tau_1d - mu))

        p_i_1d = exponential_1d * carrier_1d
        u_i_z_1d = p_i_1d * self.div_rho0_c0

        # Broadcast to full 3D field
        # shape (1, 1, N) then broadcast when used
        p_i = p_i_1d[None, None, :]
        u_i_z = u_i_z_1d[None, None, :]

        return p_i, u_i_z
