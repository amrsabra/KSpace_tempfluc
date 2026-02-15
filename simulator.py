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

        # Create spatial grid centered at 0
        self.x = (np.arange(N) - N // 2) * dx
        self.y = (np.arange(N) - N // 2) * dx
        self.z = (np.arange(N) - N // 2) * dx

        # 3D grids and radial distance from center
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.R = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)

        self.kspace_ops = KSpaceOperators(N, dx, dt) # Initialise derivatives
        self.pml = PML(N, dx) # Initialise PML boundary and components
        self.atmosphere = AtmosphereGenerator(self.X, self.Y, self.Z, self.R, self.dx) # Initialise module responsible for generating the temp field.
        self.incident = IncidentWave(self.X, self.Y, self.Z) # Initialise model that defines the analytical incident plane wave.
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
    def _simulate_scattering(self, T, n_steps=7000, fm=constants.DEFAULT_FM, tau=constants.DEFAULT_TAU, delay=constants.DEFAULT_DELAY):
        self.fm = fm
        self.tau = tau
        self.delay = delay

        # Smoothing to prevent PSTD instability at sharp edges
        T = gaussian_filter(T, sigma=1.0) # From SciPy

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
        
        # Medium properties
        rho = constants.RHO0 * constants.T0 / T 
        
        # Precompute rho Factors (1/rho0 - 1/rho) for EQN 9
        rho_const_inv = 1.0 / constants.RHO0
        rho_inv = 1.0 / rho
        corr_grad_factor = rho_const_inv - rho_inv

        print(f"Simulating {n_steps} time steps")
        start_time = time.time()

        _, u_i_z_prev = self.incident.plane_wave(-self.dt, fm, tau, delay) # Calc incident wave before t=0
        z_center_idx = self.N // 2 
        incident_power_density_sum = 0.0 # Initializing a variable to track the total power of the incident pulse.

        # Temporary buffers for summed pressure fields, this space is reused.
        # py_s and pz_s have the same shape, an array created based on px_s is perfectly sized to hold the sum of all three.
        p_s_total = np.zeros_like(px_s) 
        
        for step in range(n_steps): # Time-Stepping loop
            t = step * self.dt # Physical time

            # 1) Incident Wave Source
            p_i, u_i_z = self.incident.plane_wave(t, fm, tau, delay)
            incident_power_density_sum += p_i[0, 0, z_center_idx]**2

            duiz_dt = (u_i_z - u_i_z_prev) / self.dt
            u_i_z_prev = u_i_z

            source_term = (constants.RHO0 - rho) * duiz_dt # EQN 9
            
            # Sum pressure components for gradient calc
            p_s_total[:] = px_s + py_s + pz_s
                
            # 2) Calculate Gradients
            dpdx, dpdy, dpdz = self.kspace_ops.derivatives_xyz(p_s_total)
            
            # COME BACK TO THIS 
            rhs_ux = -dpdx * rho_const_inv + (corr_grad_factor * dpdx) 
            rhs_uy = -dpdy * rho_const_inv + (corr_grad_factor * dpdy)
            rhs_uz = -dpdz * rho_const_inv + (corr_grad_factor * dpdz) + (source_term / rho) # Source term added only to z component, because incident wave moves +z

            # 3) Update Velocities EQN 13
            self.pml.update_velocity_component(ux_s, rhs_ux, self.dt, 'x')
            self.pml.update_velocity_component(uy_s, rhs_uy, self.dt, 'y')
            self.pml.update_velocity_component(uz_s, rhs_uz, self.dt, 'z')
        
            # 4) Update Pressure
            duxdx = self.kspace_ops.derivative(ux_s, 'x') # finds spatial deriv of x-velocity
            rhs_px = -constants.RHO0 * constants.C0**2 * duxdx # Finds RHS for pressure update based on EQN 10
            self.pml.update_pressure_component(px_s, rhs_px, self.dt, 'x') # updates scattered pressure using PML
            
            duydy = self.kspace_ops.derivative(uy_s, 'y')
            rhs_py = -constants.RHO0 * constants.C0**2 * duydy
            self.pml.update_pressure_component(py_s, rhs_py, self.dt, 'y') 
            
            duzdz = self.kspace_ops.derivative(uz_s, 'z')
            rhs_pz = -constants.RHO0 * constants.C0**2 * duzdz
            self.pml.update_pressure_component(pz_s, rhs_pz, self.dt, 'z')
            
            # 5) NTFF
            p_s_total[:] = px_s + py_s + pz_s
            self.ntff.accumulate(p_s_total, ux_s, uy_s, uz_s, step)

            if (step + 1) % 100 == 0: #print steps to track progress of simulation
                print(f"  Step {step + 1}/{n_steps}")
        
        elapsed = time.time() - start_time
        print(f"Simulation completed in {elapsed:.1f} seconds")
        
        p_ff = self.ntff.compute_far_field() # Get final far-field pressure series.
        far_field = np.sum(p_ff**2, axis=1) # Calculates total scattered energy at each angle

        return far_field, self.angles_deg, incident_power_density_sum

    def simulate_scattering(self, T, n_steps=7000, fm=constants.DEFAULT_FM, tau=constants.DEFAULT_TAU, delay=constants.DEFAULT_DELAY):
        return self._simulate_scattering(T, n_steps=n_steps, fm=fm, tau=tau, delay=delay)