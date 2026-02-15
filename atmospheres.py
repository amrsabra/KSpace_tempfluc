#atmospheres.py
# Models the bragg and kolmogorov atmospheres within a tukey spherical window.

import numpy as np
from scipy.special import gamma as gamma_func
from scipy.fft import fftfreq, ifftn
import constants

class AtmosphereGenerator:
    def __init__(self, X, Y, Z, R, dx):
        # Initialising paramaters. 
        self.X = X
        self.Y = Y
        self.Z = Z
        self.R = R
        self.dx = dx
        self.N = X.shape[0]
        
    def create_tukey_window(self, r0, r1=None):
        # Spherical window with flat top and cosine taper near boundary.
        if r1 is None:
            r1 = 0.75 * r0  
        
        window = np.zeros_like(self.R)
        
        # Inside r1, the strength of the atmosphere is at maximum.
        window[self.R < r1] = 1.0
        
        # Tapered region (cosine taper) EQN 18
        # Tapered because any sharp changes in atmosphere could cause incorrect reflections. 
        taper_mask = (self.R >= r1) & (self.R <= r0)
        window[taper_mask] = 0.5 + 0.5 * np.cos(
            np.pi * (self.R[taper_mask] - r1) / (r0 - r1)
        )
        
        return window
    
    def create_bragg_atmosphere(self, fm, DT=1.0, r0=constants.R0):
        window = self.create_tukey_window(r0)
        
        # EQN 19
        T = constants.T0 + window * np.cos(4 * np.pi * fm * self.Z / constants.C0) * DT
        
        # Calculate scattering volume (after EQN 18 "by performing a volume integral of w(r)")
        # dx^3 is the volume of each cell
        # find sum of window values 0 < window < 1 and multiply it by cell volume to get total volume. 
        V_scat = np.sum(window) * self.dx**3
        
        return T, window, V_scat
    
    def create_kolmogorov_atmosphere(self, CT2, r0=constants.R0, seed=None):
        # lets user generate a specific seed if you would like for debugging and testing purposes. 
        if seed is not None:
            np.random.seed(seed)
        
        # Create k-space grid, which is the frequency domain of the original 3D grid. Because, kolm defines how much energy exists at different frequencies.
        # * 2 * np.pi because paper uses rad/m not cycle/m. normal fftfreq gives cycle/m.
        kx = fftfreq(self.N, self.dx) * 2 * np.pi
        ky = fftfreq(self.N, self.dx) * 2 * np.pi
        kz = fftfreq(self.N, self.dx) * 2 * np.pi
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = np.sqrt(KX**2 + KY**2 + KZ**2) # (kmag + k0)^-11/3
        k_mag[k_mag == 0] = 1e-10 # avoid any divison by zero and place a negligible amount
        
        # "inertial subrange", size of eddies.
        L_outer = r0 # largest possible eddy.
        L_inner = self.dx * 2 # eddy is so small it just turns into heat.
        k_outer = 2 * np.pi / L_outer # for rad/m.
        k_inner = 2 * np.pi / L_inner
        
        # exact EQN 22
        U_k = constants.KOLMOGOROV_CONSTANT * CT2 * k_mag**(-11.0 / 3.0) # (gamma function * sin(pi/3)) / 4pi^2 is in KOLMOGOROV_CONSTANT

        # enforce inertial range
        mask = (k_mag >= k_outer) & (k_mag <= k_inner)
        U_k[~mask] = 0.0 #outside range, 0.
        U_k[0, 0, 0] = 0.0

        # Generate random noise
        random_real = np.random.randn(self.N, self.N, self.N)
        random_imag = np.random.randn(self.N, self.N, self.N)
        
        # Create temperature field in k-space
        dk = 2 * np.pi / (self.N * self.dx) # k-space step, 1/total length, and 2pi for rad not cycles.
        T_k = np.sqrt(U_k * dk**3 / 2) * (random_real + 1j * random_imag) # U_k is Kolmogorov spectral density, which is temp variance (spread) per unit space. dk^3 is a volume element. We get amplitude (the amount)
        T_k[0, 0, 0] = 0 # for k = 0, T_k = inf, so its omitted. 
        
        # Transform to real space
        T_fluctuation = np.real(ifftn(T_k, workers=-1, overwrite_x=True)) # "and then applying a 3D inverse discrete fourier transform", Hargreaves et al.
        
        # Scale to realistic RMS, basically normalises fluctuations.
        T_rms_target = 1.0  # 1 K RMS
        T_rms_actual = np.std(T_fluctuation)
        if T_rms_actual > 0:
            T_fluctuation = T_fluctuation * (T_rms_target / T_rms_actual)
        
        # Apply window
        window = self.create_tukey_window(r0)
        T = constants.T0 + window * T_fluctuation
        
        # Calculate scattering volume
        V_scat = np.sum(window) * self.dx**3
        
        return T, window, V_scat