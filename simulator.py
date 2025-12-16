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

class KSpaceAcousticScattering:
    def __init__(self, N=constants.DEFAULT_N, dx=constants.DEFAULT_DX, dt=constants.DEFAULT_DT):
        self.N = N
        self.dx = dx
        self.dt = dt
        self.domain_size = self.N * self.dx

        # Create spatial grid
        self.x = (np.arange(N) - N // 2) * dx
        self.y = (np.arange(N) - N // 2) * dx
        self.z = (np.arange(N) - N // 2) * dx

        # 3D grids
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.R = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)

        self.kspace_ops = KSpaceOperators(N, dx, dt) 
        self.pml = PML(N, dx)
        self.atmosphere = AtmosphereGenerator(self.X, self.Y, self.Z, self.R, self.dx)
        self.incident = IncidentWave(self.X, self.Y, self.Z)
        self.ntff = NTFFTransform(self.x, self.y, self.z, dx,
                                  constants.C0, constants.RHO0, constants.PML_DEPTH)

        # Precompute far-field directions
        self.angles_deg = np.linspace(-180.0, 180.0, 360, endpoint=False)
        directions = np.zeros((len(self.angles_deg), 3))
        for i, angle_deg in enumerate(self.angles_deg):
            angle_rad = np.deg2rad(angle_deg)
            directions[i] = [np.sin(angle_rad), 0.0, np.cos(angle_rad)]

        self.ntff.precompute_coefficients(directions, self.dt)
        
    def create_bragg_atmosphere(self, fm, DT=1.0, r0=constants.R0):
        return self.atmosphere.create_bragg_atmosphere(fm, DT, r0)
    
    def create_kolmogorov_atmosphere(self, CT2, r0=constants.R0, seed=None):
        return self.atmosphere.create_kolmogorov_atmosphere(CT2, r0, seed)

    def _simulate_scattering(self, T, n_steps=7000, fm=constants.DEFAULT_FM, tau=constants.DEFAULT_TAU, delay=constants.DEFAULT_DELAY):
        self.fm = fm
        self.tau = tau
        self.delay = delay

        # Smoothing to prevent PSTD instability at sharp edges
        T = gaussian_filter(T, sigma=1.0)

        # NTFF setup
        self.ntff.initialize_buffer(n_steps)
        self.pml.set_dt(self.dt)

        # Fields
        px_s = np.zeros((self.N, self.N, self.N))
        py_s = np.zeros((self.N, self.N, self.N))
        pz_s = np.zeros((self.N, self.N, self.N))
        
        ux_s = np.zeros((self.N, self.N, self.N))
        uy_s = np.zeros((self.N, self.N, self.N))
        uz_s = np.zeros((self.N, self.N, self.N))
        
        ux_s_prev = np.zeros((self.N, self.N, self.N))
        uy_s_prev = np.zeros((self.N, self.N, self.N))
        uz_s_prev = np.zeros((self.N, self.N, self.N))
        
        px_s_prev = np.zeros((self.N, self.N, self.N))
        py_s_prev = np.zeros((self.N, self.N, self.N))
        pz_s_prev = np.zeros((self.N, self.N, self.N))
        
        # Medium properties
        rho = constants.RHO0 * constants.T0 / T 
        
        # Precompute Stability Factors
        rho_const_inv = 1.0 / constants.RHO0
        rho_inv = 1.0 / rho
        corr_grad_factor = rho_const_inv - rho_inv

        print(f"Simulating {n_steps} time steps")
        start_time = time.time()

        _, u_i_z_prev = self.incident.plane_wave(-self.dt, fm, tau, delay)
        z_center_idx = self.N // 2 
        incident_power_density_sum = 0.0
        
        for step in range(n_steps):
            t = step * self.dt 
            
            p_i, u_i_z = self.incident.plane_wave(t, fm, tau, delay)
            incident_power_density_sum += p_i[0, 0, z_center_idx]**2

            duiz_dt = (u_i_z - u_i_z_prev) / self.dt
            u_i_z_prev = u_i_z

            source_term = (constants.RHO0 - rho) * duiz_dt
            p_s = px_s + py_s + pz_s
                
            # --- CRITICAL STABILITY LOGIC ---
            dpdx, dpdy, dpdz = self.kspace_ops.derivatives_xyz(p_s)

            rhs_ux = -dpdx * rho_const_inv + (corr_grad_factor * dpdx)
            rhs_uy = -dpdy * rho_const_inv + (corr_grad_factor * dpdy)
            rhs_uz = -dpdz * rho_const_inv + (corr_grad_factor * dpdz) + (source_term / rho)
            # --------------------------------

            ux_s_new = self.pml.update_velocity_component(ux_s_prev, rhs_ux, self.dt, 'x')
            uy_s_new = self.pml.update_velocity_component(uy_s_prev, rhs_uy, self.dt, 'y')
            uz_s_new = self.pml.update_velocity_component(uz_s_prev, rhs_uz, self.dt, 'z')

            ux_s_prev, uy_s_prev, uz_s_prev = ux_s, uy_s, uz_s
            ux_s, uy_s, uz_s = ux_s_new, uy_s_new, uz_s_new
            
            # Pressure Update
            duxdx = self.kspace_ops.derivative(ux_s, 'x')
            rhs_px = -constants.RHO0 * constants.C0**2 * duxdx
            px_s_new = self.pml.update_pressure_component(px_s_prev, rhs_px, self.dt, 'x') 
            
            duydy = self.kspace_ops.derivative(uy_s, 'y')
            rhs_py = -constants.RHO0 * constants.C0**2 * duydy
            py_s_new = self.pml.update_pressure_component(py_s_prev, rhs_py, self.dt, 'y') 
            
            duzdz = self.kspace_ops.derivative(uz_s, 'z')
            rhs_pz = -constants.RHO0 * constants.C0**2 * duzdz
            pz_s_new = self.pml.update_pressure_component(pz_s_prev, rhs_pz, self.dt, 'z') 
            
            px_s_prev, py_s_prev, pz_s_prev = px_s, py_s, pz_s
            px_s, py_s, pz_s = px_s_new, py_s_new, pz_s_new
            
            # NTFF
            p_s_total = px_s + py_s + pz_s
            self.ntff.accumulate(p_s_total, ux_s, uy_s, uz_s, step)

            if (step + 1) % 100 == 0:
                print(f"  Step {step + 1}/{n_steps}")
        
        elapsed = time.time() - start_time
        print(f"Simulation completed in {elapsed:.1f} seconds")
        
        p_ff = self.ntff.compute_far_field() 
        far_field = np.sum(p_ff**2, axis=1)

        # FIX: use self.angles_deg instead of the undefined local variable
        return far_field, self.angles_deg, incident_power_density_sum

    def simulate_scattering(self, T, n_steps=7000, fm=constants.DEFAULT_FM, tau=constants.DEFAULT_TAU, delay=constants.DEFAULT_DELAY):
        return self._simulate_scattering(T, n_steps=n_steps, fm=fm, tau=tau, delay=delay)