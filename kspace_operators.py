# kspace_operators.py

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
import constants

class KSpaceOperators:
    def __init__(self, N, dx, dt, c0=constants.C0):
        self.N = N
        self.dx = dx
        self.dt = dt
        self.c0 = c0

        '''
        # Changing to frequency domain using wavenumbers(k) in 1D.
        # This is for more accurate results than time-domain.
        # Also, due to sinc correction term needing a wavenumber input.
        '''
        kx = 2 * np.pi * fftfreq(N, d=dx) 
        ky = 2 * np.pi * fftfreq(N, d=dx)
        kz = 2 * np.pi * fftfreq(N, d=dx)
        self.KX, self.KY, self.KZ = np.meshgrid(kx, ky, kz, indexing='ij') # Creating the 3D mesh using wavenumbers.

        k_mag = np.sqrt(self.KX**2 + self.KY**2 + self.KZ**2)
        # Avoid division by zero.
        k_mag[0, 0, 0] = 1.0 
        
        # EQN 11: Sinc correction for numerical dispersion
        # sinc(x) in numpy is sin(pi*x)/(pi*x), so we scale argument by 1/pi to match EQN 11
        self.sinc_term = np.sinc(self.c0 * self.dt * k_mag / (2.0 * np.pi)) 
        
    def derivative(self, field, axis):
        field_k = fftn(field, workers=-1) #Transforms spatial field to frequency domain.

        # Applying EQN 11. But does only 1 direction to avoid doing others if not needed and reduce computation time.
        if axis == 'x':
            deriv_k = 1j * self.KX * self.sinc_term * field_k
        elif axis == 'y':
            deriv_k = 1j * self.KY * self.sinc_term * field_k
        elif axis == 'z':
            deriv_k = 1j * self.KZ * self.sinc_term * field_k
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        deriv = ifftn(deriv_k, workers=-1) # Back to spatial domain
        return np.real(deriv) 

    '''
    # For dp/dx, dp/dy, and dp/dz, 
    # this is an optimatization method were we do 1 forward FFT and 3 inverse FFT 
    # instead of 3 forward and 3 backward for faster computation.
    '''    
    def derivatives_xyz(self, field): 
        field_k = fftn(field, workers=-1)

        deriv_x_k = 1j * self.KX * self.sinc_term * field_k
        deriv_y_k = 1j * self.KY * self.sinc_term * field_k
        deriv_z_k = 1j * self.KZ * self.sinc_term * field_k

        dx = np.real(ifftn(deriv_x_k, workers=-1))
        dy = np.real(ifftn(deriv_y_k, workers=-1))
        dz = np.real(ifftn(deriv_z_k, workers=-1))
        return dx, dy, dz