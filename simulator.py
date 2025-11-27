# simulator.py

import numpy as np
import time
import constants
from kspace_operators import KSpaceOperators
from pml import PML
from atmospheres import AtmosphereGenerator
from incident_wave import IncidentWave
from ntff import NTFFTransform

class KSpaceAcousticScattering:
    def __init__(self, N=constants.DEFAULT_N, dx=constants.DEFAULT_DX, dt=constants.DEFAULT_DT):
        self.N = N
        self.dx = dx
        self.dt = dt

        # Domain parameters
        self.domain_size = self.N * self.dx
        self.x = np.linspace(-self.domain_size/2, self.domain_size/2, self.N) # returns [-128,-127,...,127,128]
        self.y = np.linspace(-self.domain_size/2, self.domain_size/2, self.N)
        self.z = np.linspace(-self.domain_size/2, self.domain_size/2, self.N)
        
        # Create 3D grids
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.R = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        
        # Initialize components
        self.kspace_ops = KSpaceOperators(self.N, self.dx, self.dt, constants.C0)
        self.pml = PML(self.N, self.dx, constants.C0, constants.PML_DEPTH, constants.PML_ABSORPTION)
        self.atmosphere = AtmosphereGenerator(self.X, self.Y, self.Z, self.R, self.dx)
        self.incident = IncidentWave(self.X, self.Y, self.Z)
        self.ntff = NTFFTransform(self.x, self.y, self.z, self.dx, constants.C0, constants.RHO0, constants.PML_DEPTH)
        
        
    def create_bragg_atmosphere(self, fm, DT=1.0, r0=constants.R0):
        return self.atmosphere.create_bragg_atmosphere(fm, DT, r0)
    
    def create_kolmogorov_atmosphere(self, CT2, r0=constants.R0, seed=None):
        return self.atmosphere.create_kolmogorov_atmosphere(CT2, r0, seed)

    def _simulate_scattering(self, T, n_steps=7000, fm=constants.DEFAULT_FM, tau=constants.DEFAULT_TAU, delay=constants.DEFAULT_DELAY):
        self.fm = fm
        self.tau = tau
        self.delay = delay

        # Far-field directions
        angles_deg = np.linspace(-180, 180, 360, endpoint=False)
        directions = np.zeros((len(angles_deg), 3))
        for i, angle_deg in enumerate(angles_deg): # i is an array with three components declared in line before, NTFF needs direction split into components to project velocity onto direction d
            angle_rad = np.deg2rad(angle_deg) # np expects rad not deg.
            directions[i] = [np.sin(angle_rad), 0.0, np.cos(angle_rad)] # "far-field directions were modeled spaced at equal angles over the y = 0 plane."
        
        self.ntff.precompute_coefficients(directions, self.dt) # precomputes all the geometric factors and time weights in EQN 15.
        self.ntff.initialize_buffer(n_steps) # Creates np.zeros array of dir*nsteps
        self.pml.set_dt(self.dt) # Tell the PML object what the time step is.

        # Initialize scattered fields (split into directional components for PML)
        # Current time step fields
        px_s = np.zeros((self.N, self.N, self.N))
        py_s = np.zeros((self.N, self.N, self.N))
        pz_s = np.zeros((self.N, self.N, self.N))
        
        ux_s = np.zeros((self.N, self.N, self.N))
        uy_s = np.zeros((self.N, self.N, self.N))
        uz_s = np.zeros((self.N, self.N, self.N))
        
        # Previous time step fields (for staggered grid)
        ux_s_prev = np.zeros((self.N, self.N, self.N))
        uy_s_prev = np.zeros((self.N, self.N, self.N))
        uz_s_prev = np.zeros((self.N, self.N, self.N))
        
        px_s_prev = np.zeros((self.N, self.N, self.N))
        py_s_prev = np.zeros((self.N, self.N, self.N))
        pz_s_prev = np.zeros((self.N, self.N, self.N))
        
        # Medium properties
        rho = constants.RHO0 * constants.T0 / T # ideal gas equation

        print(f"Simulating {n_steps} time steps")
        start_time = time.time()

        for step in range(n_steps):
            # Every update (incident field, source term, NTFF) depends on the physical time, not just the index.
            t = step * self.dt # its like frame number * number of frames; it's the timestamp in the simulation
            
            # Incident field and source term (EQN 9)
            p_i, u_i_z = self.incident.plane_wave(t, fm, tau, delay)
            p_i_prev, u_i_z_prev = self.incident.plane_wave(t - self.dt, fm, tau, delay)
            duiz_dt = (u_i_z - u_i_z_prev) / self.dt
            source_term = (constants.RHO0 - rho) * duiz_dt # EQN 9
            
            p_s = px_s + py_s + pz_s
            
            # Update velocity (EQN 9 with PML from EQN 13)
            dpdx = self.kspace_ops.derivative(p_s, 'x')
            rhs_ux = -dpdx / rho
            ux_s_new = self.pml.update_velocity_component(ux_s_prev, rhs_ux, self.dt, 'x')
            
            dpdy = self.kspace_ops.derivative(p_s, 'y')
            rhs_uy = -dpdy / rho
            uy_s_new = self.pml.update_velocity_component(uy_s_prev, rhs_uy, self.dt, 'y')
            
            dpdz = self.kspace_ops.derivative(p_s, 'z')
            rhs_uz = -dpdz / rho + source_term / rho
            uz_s_new = self.pml.update_velocity_component(uz_s_prev, rhs_uz, self.dt, 'z')
            
            ux_s_prev, uy_s_prev, uz_s_prev = ux_s, uy_s, uz_s
            ux_s, uy_s, uz_s = ux_s_new, uy_s_new, uz_s_new
            
            # Update pressure (EQN 10 with PML from EQN 13)
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
            
            # Accumulate far-field data for later (EQN 17)
            p_s_total = px_s + py_s + pz_s
            self.ntff.accumulate(p_s_total, ux_s, uy_s, uz_s, step)

            if (step + 1) % 100 == 0: # every 100 steps, print progress
                print(f"  Step {step + 1}/{n_steps}")
        
        elapsed = time.time() - start_time
        print(f"Simulation completed in {elapsed:.1f} seconds")
        
        # Compute final far-field
        print("Computing far-field from NTFF buffer...")
        p_ff = self.ntff.compute_far_field() # takes all near field data and computes far field using EQN 17.
        far_field = np.sum(p_ff**2, axis=1) #integrates energy of far-field over time.

        return far_field, angles_deg

    def simulate_scattering(self, T, n_steps=7000, fm=constants.DEFAULT_FM, tau=constants.DEFAULT_TAU, delay=constants.DEFAULT_DELAY):
        return self._simulate_scattering(T, n_steps=n_steps, fm=fm, tau=tau, delay=delay)
    
    def calculate_scattering_cross_section(self, far_field):
        H = far_field / np.max(far_field)
        H_dB = 10 * np.log10(H + 1e-12)
        return H_dB
