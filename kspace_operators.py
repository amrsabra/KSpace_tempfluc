#kspace_operators.py

import numpy as np
from scipy.fft import fftn, ifftn, fft, ifft, fftfreq
import constants

class KSpaceOperators:
    def __init__(self, N, dx, dt, c0=constants.C0):
        self.N = N
        self.dx = dx
        self.dt = dt
        self.c0 = c0

        kx = 2 * np.pi * fftfreq(N, d=dx)
        ky = 2 * np.pi * fftfreq(N, d=dx)
        kz = 2 * np.pi * fftfreq(N, d=dx)
        self.KX, self.KY, self.KZ = np.meshgrid(kx, ky, kz, indexing='ij')

        k_mag = np.sqrt(self.KX**2 + self.KY**2 + self.KZ**2)
        k_mag[0, 0, 0] = 1.0
        self.sinc_term = np.sinc(self.c0 * self.dt * k_mag / (2.0 * np.pi)) # eqn 11
        
    def derivative(self, field, axis):
        field_k = fftn(field, workers=-1, overwrite_x=True)

        if axis == 'x':
            deriv_k = 1j * self.KX * self.sinc_term * field_k
        elif axis == 'y':
            deriv_k = 1j * self.KY * self.sinc_term * field_k
        elif axis == 'z':
            deriv_k = 1j * self.KZ * self.sinc_term * field_k
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        deriv = ifftn(deriv_k, workers=-1, overwrite_x=True)
        return np.real(deriv)
    
    def derivatives_xyz(self, field):
        """Compute dp/dx, dp/dy, dp/dz sharing a single FFT of field."""
        field_k = fftn(field, workers=-1, overwrite_x=True)

        deriv_x_k = 1j * self.KX * self.sinc_term * field_k
        deriv_y_k = 1j * self.KY * self.sinc_term * field_k
        deriv_z_k = 1j * self.KZ * self.sinc_term * field_k

        dx = np.real(ifftn(deriv_x_k, workers=-1, overwrite_x=True))
        dy = np.real(ifftn(deriv_y_k, workers=-1, overwrite_x=True))
        dz = np.real(ifftn(deriv_z_k, workers=-1, overwrite_x=True))
        return dx, dy, dz