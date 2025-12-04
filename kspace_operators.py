#kspace_operators.py

import numpy as np
from scipy.fft import fftn, ifftn, fft, ifft, fftfreq

class KSpaceOperators:
    def __init__(self, N, dx):
        self.N = N
        self.dx = dx

        kx = 2 * np.pi * fftfreq(N, d=dx)
        ky = 2 * np.pi * fftfreq(N, d=dx)
        kz = 2 * np.pi * fftfreq(N, d=dx)
        self.KX, self.KY, self.KZ = np.meshgrid(kx, ky, kz, indexing='ij')

        k_mag = np.sqrt(self.KX**2 + self.KY**2 + self.KZ**2)
        k_mag[k_mag == 0] = 1.0
        self.sinc_term = np.sinc(k_mag * dx / (2 * np.pi))
        
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
        field_k = fftn(field)

        deriv_x_k = 1j * self.KX * self.sinc_term * field_k
        deriv_y_k = 1j * self.KY * self.sinc_term * field_k
        deriv_z_k = 1j * self.KZ * self.sinc_term * field_k

        dx = np.real(ifftn(deriv_x_k, workers=-1, overwrite_x=True))
        dy = np.real(ifftn(deriv_y_k, workers=-1, overwrite_x=True))
        dz = np.real(ifftn(deriv_z_k, workers=-1, overwrite_x=True))
        return dx, dy, dz