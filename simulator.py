# simulator.py

import numpy as np
import time
import constants
from kspace_operators import KSpaceOperators
from pml import PML
from atmospheres import AtmosphereGenerator
from incident_wave import IncidentWave
from ntff import NTFFTransform
from scipy.ndimage import gaussian_filter
import scipy.fft as sfft

class KSpaceAcousticScattering:
    def __init__(self, N=constants.DEFAULT_N, dx=constants.DEFAULT_DX, dt=constants.DEFAULT_DT):
        self.N = N
        self.dx = dx
        self.dt = dt
        self.domain_size = self.N * self.dx

        # Create spatial grid centered at 0
        self.x = (np.arange(N) - N // 2) * dx
        self.y = (np.arange(N) - N // 2) * dx
        self.z = (np.arange(N) - N // 2) * dx

        # 3D grids and radial distance from center
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.R = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)

        # Calculate Wind Vector components from constants
        angle_rad = np.deg2rad(constants.WIND_DIR_DEG)
        self.vx = constants.WIND_SPEED * np.sin(angle_rad)
        self.vy = 0.0 # Assuming 2D wind in X-Z plane for simplicity
        self.vz = constants.WIND_SPEED * np.cos(angle_rad)

        pml_d = constants.PML_DEPTH
        wind_mask_1d = np.ones(self.N)
        
        # Smooth cosine taper for the PML region (0 at edge, 1 in the domain)
        taper_curve = 0.5 * (1 - np.cos(np.pi * np.arange(pml_d) / pml_d))
        wind_mask_1d[:pml_d] = taper_curve
        wind_mask_1d[-pml_d:] = taper_curve[::-1]
        
        # Create 3D mask
        WM_X, WM_Y, WM_Z = np.meshgrid(wind_mask_1d, wind_mask_1d, wind_mask_1d, indexing="ij")
        
        # 3D wind velocity arrays
        self.vx_3d = self.vx * WM_X * WM_Y * WM_Z
        self.vz_3d = self.vz * WM_X * WM_Y * WM_Z

        self.kspace_ops = KSpaceOperators(N, dx, dt) # Initialise derivatives
        self.pml = PML(N, dx) # Initialise PML boundary and components
        self.atmosphere = AtmosphereGenerator(self.X, self.Y, self.Z, self.R, self.dx) # Initialise module responsible for generating the temp field.
        self.incident = IncidentWave(self.X, self.Y, self.Z, wind_vz=self.vz) # Initialise model that defines the analytical incident plane wave.
        self.ntff = NTFFTransform(self.x, self.y, self.z, dx,
                                  constants.C0, constants.RHO0, constants.PML_DEPTH) # Initialise NTFF module.

        # Precompute far-field directions
        self.angles_deg = np.linspace(-180.0, 180.0, 360, endpoint=False) # Defines 360 observation angles for far field scattering 
        directions = np.zeros((len(self.angles_deg), 3)) #3D vector array for each observer.
        for i, angle_deg in enumerate(self.angles_deg):
            angle_rad = np.deg2rad(angle_deg)
            directions[i] = [np.sin(angle_rad), 0.0, np.cos(angle_rad)]

        self.ntff.precompute_coefficients(directions, self.dt) # Passes the directions and time step to the NTFF module for preparation.
        
    def create_bragg_atmosphere(self, fm, DT=1.0, r0=constants.R0): 
        return self.atmosphere.create_bragg_atmosphere(fm, DT, r0)
    
    def create_kolmogorov_atmosphere(self, CT2, r0=constants.R0, seed=None):
        return self.atmosphere.create_kolmogorov_atmosphere(CT2, r0, seed)

    # This is what runs the actual time-stepping loop to find scattering data.
    def _simulate_scattering(self, T_data, n_steps=7000, fm=constants.DEFAULT_FM, tau=constants.DEFAULT_TAU, delay=constants.DEFAULT_DELAY):
        self.fm = fm
        self.tau = tau
        self.delay = delay

        if isinstance(T_data, tuple):
            T_init, window, V_scat, T_k, window_mask = T_data
            drifting = True
            T = T_init
        else:
            T = T_data
            drifting = False

        # Smoothing to prevent PSTD instability at sharp edges
        # Sigma increased to 2 form 1 to stabilize high-frequency advection
        T = gaussian_filter(T, sigma=0.5) # From SciPy

        # NTFF setup
        self.ntff.initialize_buffer(n_steps) # Allocates memory 360degrees x n_steps, to store results over the during of the sim.
        self.pml.set_dt(self.dt) # Sets deltat for PML damping.

        # Fields (Basically 256 256x256 arrays filled with zeros for each component, to store pressure and velocity values)
        px_s = np.zeros((self.N, self.N, self.N))
        py_s = np.zeros((self.N, self.N, self.N))
        pz_s = np.zeros((self.N, self.N, self.N))
        
        ux_s = np.zeros((self.N, self.N, self.N))
        uy_s = np.zeros((self.N, self.N, self.N))
        uz_s = np.zeros((self.N, self.N, self.N))

        # Precompute phase shift per timestep for environmental drift
        if drifting:
            phase_shift_step = np.exp(-1j * (self.kspace_ops.KX * self.vx + 
                                             self.kspace_ops.KY * self.vy + 
                                             self.kspace_ops.KZ * self.vz) * self.dt)
        
        # Precompute rho Factors (1/rho0 - 1/rho) for EQN 9
        rho_const_inv = 1.0 / constants.RHO0
        
        # Calculate initial properties before loop
        rho = constants.RHO0 * constants.T0 / T 
        corr_grad_factor = rho_const_inv - (1.0 / rho)

        print(f"Simulating {n_steps} time steps")
        start_time = time.time()

        _, u_i_z_prev = self.incident.plane_wave(-self.dt, fm, tau, delay) # Calc incident wave before t=0
        z_center_idx = self.N // 2 
        incident_power_density_sum = 0.0 # Initializing a variable to track the total power of the incident pulse.

        # Temporary buffers for summed pressure fields, this space is reused.
        # py_s and pz_s have the same shape, an array created based on px_s is perfectly sized to hold the sum of all three.
        p_s_total = np.zeros_like(px_s) 

        # Pre-allocate gradient buffers to prevent memory re-allocation inside the loop
        dpdx, dpdy, dpdz = [np.zeros((self.N, self.N, self.N)) for _ in range(3)]
        duxdx, duxdy, duxdz = [np.zeros((self.N, self.N, self.N)) for _ in range(3)]
        duydx, duydy, duydz = [np.zeros((self.N, self.N, self.N)) for _ in range(3)]
        duzdx, duzdy, duzdz = [np.zeros((self.N, self.N, self.N)) for _ in range(3)]

        dpdx_adv, dpdz_adv = [np.zeros((self.N, self.N, self.N)) for _ in range(2)]
        duxdx_adv, duxdz_adv = [np.zeros((self.N, self.N, self.N)) for _ in range(2)]
        duydx_adv, duydz_adv = [np.zeros((self.N, self.N, self.N)) for _ in range(2)]
        duzdx_adv, duzdz_adv = [np.zeros((self.N, self.N, self.N)) for _ in range(2)]

        # New way of measuring the pressure for wind speed
        self.sensor_fwd_raw = np.zeros(n_steps)
        self.sensor_bwd_raw = np.zeros(n_steps)
        
        # Calculate indices for 0, 0, +2.4m and 0, 0, -2.4m
        center_idx = self.N // 2
        sensor_offset_idx = int(2.2 / self.dx)
        
        idx_top = center_idx + sensor_offset_idx  # 0° (Forward)
        idx_bot = center_idx - sensor_offset_idx  # 180° (Backward)

        for step in range(n_steps): # Time-Stepping loop
            t = step * self.dt # Physical time

        # Environment Drift (Fluctuation) - Now continuous for numerical stability
            if drifting and step > 0:
                # Apply phase shift every single step for smoothness
                T_k *= phase_shift_step
                T_fluct = self.kspace_ops.ifft_drift(T_k) # Uses the optimized C-path
                T = constants.T0 + window_mask * T_fluct
                
                # Recalculate medium properties
                rho = constants.RHO0 * constants.T0 / T 
                corr_grad_factor = rho_const_inv - (1.0 / rho)

            # 1) Incident Wave Source
            p_i, u_i_z = self.incident.plane_wave(t, fm, tau, delay)
            incident_power_density_sum += p_i[0, 0, z_center_idx]**2

            duiz_dt = (u_i_z - u_i_z_prev) / self.dt
            u_i_z_prev = u_i_z

            source_term = (constants.RHO0 - rho) * duiz_dt # EQN 9
            
            # Sum pressure components for gradient calc
            p_s_total[:] = px_s + py_s + pz_s

            # record the real physical pressure
            self.sensor_fwd_raw[step] = p_s_total[center_idx, center_idx, idx_top]
            self.sensor_bwd_raw[step] = p_s_total[center_idx, center_idx, idx_bot]
                
            # 2a) Calculate Advection Gradients for Pressure (no sinc correction)
            self.kspace_ops.advection_derivatives_xz(p_s_total, dx=dpdx_adv, dz=dpdz_adv)
            
            # Pure Spectral Advection for Pressure
            p_adv = self.vx_3d * dpdx_adv + self.vz_3d * dpdz_adv

            # 2b) Calculate Acoustic Gradients for Pressure (WITH sinc correction)
            self.kspace_ops.derivatives_xyz(p_s_total, dx=dpdx, dy=None, dz=dpdz)
            
            # 2c) Calculate Velocity Advection (second order to prevent damping)
            def upwind(field):
                adv = np.zeros_like(field)
                
                # X-Direction Upwinding
                if self.vx > 0: 
                    adv += self.vx_3d * (3*field - 4*np.roll(field, 1, axis=0) + np.roll(field, 2, axis=0)) / (2 * self.dx)
                elif self.vx < 0: 
                    adv += self.vx_3d * (-np.roll(field, -2, axis=0) + 4*np.roll(field, -1, axis=0) - 3*field) / (2 * self.dx)
                    
                # Z-Direction Upwinding
                if self.vz > 0: 
                    adv += self.vz_3d * (3*field - 4*np.roll(field, 1, axis=2) + np.roll(field, 2, axis=2)) / (2 * self.dx)
                elif self.vz < 0: 
                    adv += self.vz_3d * (-np.roll(field, -2, axis=2) + 4*np.roll(field, -1, axis=2) - 3*field) / (2 * self.dx)
                    
                return adv

            u_adv_x = upwind(ux_s)
            u_adv_y = upwind(uy_s)
            u_adv_z = upwind(uz_s)
            
            # 3) Update Velocities EQN 13 (Now including advection terms)
            rhs_ux = -dpdx * rho_const_inv + (corr_grad_factor * dpdx) - u_adv_x
            rhs_uy = -dpdy * rho_const_inv + (corr_grad_factor * dpdy) - u_adv_y
            rhs_uz = -dpdz * rho_const_inv + (corr_grad_factor * dpdz) + (source_term / rho) - u_adv_z

            self.pml.update_velocity_component(ux_s, rhs_ux, self.dt, 'x')
            self.pml.update_velocity_component(uy_s, rhs_uy, self.dt, 'y')
            self.pml.update_velocity_component(uz_s, rhs_uz, self.dt, 'z')
        
            # We calculate velocity gradients AFTER the velocities are updated.
            self.kspace_ops.derivatives_xyz(ux_s, dx=duxdx, dy=None, dz=duxdz)
            self.kspace_ops.derivatives_xyz(uy_s, dx=duydx, dy=duydy, dz=duydz)
            self.kspace_ops.derivatives_xyz(uz_s, dx=duzdx, dy=None, dz=duzdz)

            # 4) Update Pressure (Now including advection terms)
            rhs_px = -constants.RHO0 * constants.C0**2 * duxdx - (p_adv / 3.0)
            self.pml.update_pressure_component(px_s, rhs_px, self.dt, 'x') 
            
            rhs_py = -constants.RHO0 * constants.C0**2 * duydy - (p_adv / 3.0)
            self.pml.update_pressure_component(py_s, rhs_py, self.dt, 'y') 
            
            rhs_pz = -constants.RHO0 * constants.C0**2 * duzdz - (p_adv / 3.0)
            self.pml.update_pressure_component(pz_s, rhs_pz, self.dt, 'z')
            
            # 5) NTFF
            p_s_total[:] = px_s + py_s + pz_s
            self.ntff.accumulate(p_s_total, ux_s, uy_s, uz_s, step)

            # MONITORING: Print max pressure to see if it is blowing up
            if (step + 1) % 100 == 0:
                max_p = np.max(np.abs(p_s_total))
                print(f"  Step {step + 1}/{n_steps} | Max Pressure: {max_p:.4e}", flush=True)
                if max_p > 1e10:
                    print("\n!!! INSTABILITY DETECTED: Pressure is blowing up. Terminating test. !!!")
                    raise FloatingPointError("Numerical instability detected.")
        
        elapsed = time.time() - start_time
        print(f"Simulation completed in {elapsed:.1f} seconds")
        
        p_ff = self.ntff.compute_far_field() # Get final far-field pressure series.
        far_field = np.sum(p_ff**2, axis=1) # Calculates total scattered energy at each angle

        return far_field, self.angles_deg, incident_power_density_sum, self.sensor_fwd_raw, self.sensor_bwd_raw

    def simulate_scattering(self, T, n_steps=7000, fm=constants.DEFAULT_FM, tau=constants.DEFAULT_TAU, delay=constants.DEFAULT_DELAY):
        return self._simulate_scattering(T, n_steps=n_steps, fm=fm, tau=tau, delay=delay)