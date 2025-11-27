#kspace_operators.py

import numpy as np

class KSpaceOperators:
    def __init__(self, N, dx, dt, c0):
        self.N = N
        self.dx = dx
        self.dt = dt
        self.c0 = c0
        
        self._init_kspace_grid()
        
    def _init_kspace_grid(self): # prepares all k space data that is constant in time
        # 1D freq vectors but with 2pi to change unit from cycle/m to rad/m; k = 2pif
        kx = 2 * np.pi * np.fft.fftfreq(self.N, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.N, self.dx)
        kz = 2 * np.pi * np.fft.fftfreq(self.N, self.dx)
        
        # Transform 1D array vectors to a 3D array mesh
        self.KX, self.KY, self.KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        #Kmod
        self.K = np.sqrt(self.KX**2 + self.KY**2 + self.KZ**2)
        
        # /pi because in np, its sinc(pi*x)
        self.sinc_term = np.sinc(self.c0 * self.dt * self.K / (2 * np.pi))
        
    def derivative(self, field, direction): # field is the 3D np array of what we want to find the derivative of.
        # Transform to k-space (complex)
        field_k = np.fft.fftn(field) # N-dimensional discrete Fourier Transform, returns complex values.
        
        # split components up
        if direction == 'x':
            k_component = self.KX
        elif direction == 'y':
            k_component = self.KY
        elif direction == 'z':
            k_component = self.KZ
        else:
            raise ValueError("Direction must be 'x', 'y', or 'z'")
        
        # We used 1j (imaginary), because adding j would cause a +90 degree shift which would transform isin(kx) to cos(kx)
        deriv_k = 1j * k_component * self.sinc_term * field_k
        
        # Transform back to real space
        deriv = np.real(np.fft.ifftn(deriv_k)) # EQN 11
        
        return deriv