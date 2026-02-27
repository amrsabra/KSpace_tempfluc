# kspace_operators.py

import numpy as np
import pyfftw
import multiprocessing
from scipy.fft import fftfreq
import constants

pyfftw.interfaces.cache.enable()

class KSpaceOperators:
    def __init__(self, N, dx, dt, c0=constants.C0):
        self.N = N
        self.dx = dx
        self.dt = dt
        self.c0 = c0

        # Define self.threads before creating FFTW plans.
        self.threads = multiprocessing.cpu_count()

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

        k_max = np.pi / self.dx # The maximum physical frequency the grid can hold (Nyquist)

        
        # EQN 11: Sinc correction for numerical dispersion
        # sinc(x) in numpy is sin(pi*x)/(pi*x), so we scale argument by 1/pi to match EQN 11
        self.sinc_term = np.sinc(self.c0 * self.dt * k_mag / (2.0 * np.pi)) 

        # Existing acoustic operators (with sinc)
        self.operator_x = 1j * self.KX * self.sinc_term
        self.operator_y = 1j * self.KY * self.sinc_term
        self.operator_z = 1j * self.KZ * self.sinc_term

        # Advection operators
        self.adv_operator_x = 1j * self.KX
        self.adv_operator_z = 1j * self.KZ

        # FFTW array preparation
        self.in_array = pyfftw.empty_aligned((N, N, N), dtype='complex128')
        self.k_array = pyfftw.empty_aligned((N, N, N), dtype='complex128')
        self.out_array = pyfftw.empty_aligned((N, N, N), dtype='complex128')

        # Create the Forward Plan (FFTN)
        self.forward_fft = pyfftw.FFTW(
            self.in_array, self.k_array, 
            axes=(0, 1, 2), direction='FFTW_FORWARD', 
            flags=('FFTW_MEASURE',), threads=self.threads
        )

        # 3. Create the Backward Plan (IFFTN)
        self.backward_fft = pyfftw.FFTW(
            self.k_array, self.out_array, 
            axes=(0, 1, 2), direction='FFTW_BACKWARD', 
            flags=('FFTW_MEASURE',), threads=self.threads
        )

        # Backward Plan for Wind advection
        self.drift_out = pyfftw.empty_aligned((N, N, N), dtype='complex128')
        self.drift_ifft = pyfftw.FFTW(
            self.k_array, self.drift_out, 
            axes=(0, 1, 2), direction='FFTW_BACKWARD', 
            flags=('FFTW_MEASURE',), threads=self.threads
        )

        # internal k-space buffer to reuse the same memory blocks for derivatives_xyz
        self.field_k_buffer = pyfftw.empty_aligned((N, N, N), dtype='complex128')

    def ifft_drift(self, T_k):
        self.k_array[:] = T_k
        self.drift_ifft()
        return np.real(self.drift_out)
        
    def derivative(self, field, axis):
        #Transforms spatial field to frequency domain.
        self.in_array[:] = field
        self.forward_fft() 

        # Applying EQN 11. But does only 1 direction to avoid doing others if not needed and reduce computation time.
        if axis == 'x':
            self.k_array[:] *= self.operator_x
        elif axis == 'y':
            self.k_array[:] *= self.operator_y
        elif axis == 'z':
            self.k_array[:] *= self.operator_z
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        self.backward_fft()
        return np.real(self.out_array)

    '''
    # For dp/dx, dp/dy, and dp/dz, 
    # this is an optimatization method were we do 1 forward FFT and 3 inverse FFT 
    # instead of 3 forward and 3 backward for faster computation.
    '''    
    def derivatives_xyz(self, field, dx=None, dy=None, dz=None): 
        # "in-place" computation
        self.in_array[:] = field
        self.forward_fft()
        self.field_k_buffer[:] = self.k_array # Store the k-space field in pre-allocated buffer

        # 2. X-Derivative calculation
        if dx is not None:
            self.k_array[:] = self.field_k_buffer * self.operator_x
            self.backward_fft()
            dx[:] = np.real(self.out_array)

        # 3. Y-Derivative calculation
        if dy is not None:
            self.k_array[:] = self.field_k_buffer * self.operator_y
            self.backward_fft()
            dy[:] = np.real(self.out_array)

        # 4. Z-Derivative calculation
        if dz is not None:
            self.k_array[:] = self.field_k_buffer * self.operator_z
            self.backward_fft()
            dz[:] = np.real(self.out_array)

        return dx, dy, dz

    def advection_derivatives_xz(self, field, dx, dz): 
        """Calculates exact spatial derivatives without acoustic sinc correction for wind advection."""
        self.in_array[:] = field
        self.forward_fft()
        self.field_k_buffer[:] = self.k_array

        # X-Derivative calculation
        self.k_array[:] = self.field_k_buffer * self.adv_operator_x
        self.backward_fft()
        dx[:] = np.real(self.out_array)

        # Z-Derivative calculation
        self.k_array[:] = self.field_k_buffer * self.adv_operator_z
        self.backward_fft()
        dz[:] = np.real(self.out_array)