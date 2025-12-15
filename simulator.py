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
        self.domain_size = self.N * self.dx

        # Create spatial grid
        self.x = (np.arange(N) - N // 2) * dx
        self.y = (np.arange(N) - N // 2) * dx
        self.z = (np.arange(N) - N // 2) * dx

        # 3D grids
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.R = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)

        # Operators, PML, incident field, NTFF
        # FIX A: Removed 'dt' argument as per kspace_operators.py signature
        self.kspace_ops = KSpaceOperators(N, dx) 
        self.pml = PML(N, dx)
        self.atmosphere = AtmosphereGenerator(self.X, self.Y, self.Z, self.R, self.dx)
        self.incident = IncidentWave(self.X, self.Y, self.Z)
        self.ntff = NTFFTransform(self.x, self.y, self.z, dx,
                                  constants.C0, constants.RHO0, constants.PML_DEPTH)

        # Precompute far-field directions once
        self.angles_deg = np.linspace(-180.0, 180.0, 360, endpoint=False)
        directions = np.zeros((len(self.angles_deg), 3))
        for i, angle_deg in enumerate(self.angles_deg):
            angle_rad = np.deg2rad(angle_deg)
            directions[i] = [np.sin(angle_rad), 0.0, np.cos(angle_rad)]

        # Precompute NTFF coefficients once for this grid and dt
        self.ntff.precompute_coefficients(directions, self.dt)
        
        
    def create_bragg_atmosphere(self, fm, DT=1.0, r0=constants.R0):
        return self.atmosphere.create_bragg_atmosphere(fm, DT, r0)
    
    def create_kolmogorov_atmosphere(self, CT2, r0=constants.R0, seed=None):
        return self.atmosphere.create_kolmogorov_atmosphere(CT2, r0, seed)

    def _simulate_scattering(self, T, n_steps=7000, fm=constants.DEFAULT_FM, tau=constants.DEFAULT_TAU, delay=constants.DEFAULT_DELAY):
        self.fm = fm
        self.tau = tau
        self.delay = delay

        # Directions and coefficients already precomputed in __init__
        angles_deg = self.angles_deg

        # NTFF time buffer for this run
        self.ntff.initialize_buffer(n_steps)
        self.pml.set_dt(self.dt)

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

        # Cache previous incident velocity once
        _, u_i_z_prev = self.incident.plane_wave(-self.dt, fm, tau, delay)

        # FIX B START: Initialize accumulator for Incident Power Density (Denominator in Eq 20)
        # We sample the center z-slice of the incident plane wave.
        z_center_idx = self.N // 2 
        incident_power_density_sum = 0.0
        # FIX B END
        
        for step in range(n_steps):
            # Every update (incident field, source term, NTFF) depends on the physical time, not just the index.
            t = step * self.dt 
            
            # Incident field and source term (EQN 9)
            p_i, u_i_z = self.incident.plane_wave(t, fm, tau, delay)

            # FIX B START: Accumulate incident power density
            incident_power_density_sum += p_i[0, 0, z_center_idx]**2
            # FIX B END

            duiz_dt = (u_i_z - u_i_z_prev) / self.dt
            u_i_z_prev = u_i_z

            source_term = (constants.RHO0 - rho) * duiz_dt # EQN 9
            
            p_s = px_s + py_s + pz_s
                
            # Velocity update (Eq 9 + PML), using shared FFT for dp/dx,dp/dy,dp/dz
            dpdx, dpdy, dpdz = self.kspace_ops.derivatives_xyz(p_s)

            # --- START STABILITY FIX (YOUR CORRECTION) ---
            rho_const_inv = 1.0 / constants.RHO0  # 1/rho0 (constant)
            rho_inv = 1.0 / rho                   # 1/rho(r) (spatially varying)

            # The gradient correction factor: (1/rho0 - 1/rho(r))
            corr_grad_factor = rho_const_inv - rho_inv 

            # 1. K-Space Base Term (Homogeneous: -grad(ps)/rho0)
            rhs_ux_base = -dpdx * rho_const_inv
            rhs_uy_base = -dpdy * rho_const_inv
            rhs_uz_base = -dpdz * rho_const_inv

            # 2. Correction Term (Explicit Source: (1/rho0 - 1/rho(r)) * grad(ps))
            corr_grad_ux = corr_grad_factor * dpdx
            corr_grad_uy = corr_grad_factor * dpdy
            corr_grad_uz = corr_grad_factor * dpdz

            # 3. Incident Source Term: S_i_z = source_term / rho
            source_term_z = source_term / rho

            # 4. Total RHS = Base Term + Correction Term + Incident Source Term
            rhs_ux = rhs_ux_base + corr_grad_ux
            rhs_uy = rhs_uy_base + corr_grad_uy
            rhs_uz = rhs_uz_base + corr_grad_uz + source_term_z
            # --- END STABILITY FIX ---

            ux_s_new = self.pml.update_velocity_component(ux_s_prev, rhs_ux, self.dt, 'x')
            uy_s_new = self.pml.update_velocity_component(uy_s_prev, rhs_uy, self.dt, 'y')
            uz_s_new = self.pml.update_velocity_component(uz_s_prev, rhs_uz, self.dt, 'z')

            ux_s_prev, uy_s_prev, uz_s_prev = ux_s, uy_s, uz_s
            ux_s, uy_s, uz_s = ux_s_new, uy_s_new, uz_s_new
            
            # Update pressure (EQN 10 with PML from EQN 13)
            # Note: Eq 10 has no heterogeneity or correction term because rho(r)c^2(r) = rho0*c0^2
            duxdx = self.kspace_ops.derivative(ux_s, 'x')
            rhs_px = -constants.RHO0 * constants.C0**2 * duxdx
            px_s_new = self.pml.update_pressure_component(px_s_prev, rhs_px, self.dt, 'x') # add the PML damping to RHS
            
            duydy = self.kspace_ops.derivative(uy_s, 'y')
            rhs_py = -constants.RHO0 * constants.C0**2 * duydy
            py_s_new = self.pml.update_pressure_component(py_s_prev, rhs_py, self.dt, 'y') # add the PML damping to RHS
            
            duzdz = self.kspace_ops.derivative(uz_s, 'z')
            rhs_pz = -constants.RHO0 * constants.C0**2 * duzdz
            pz_s_new = self.pml.update_pressure_component(pz_s_prev, rhs_pz, self.dt, 'z') # add the PML damping to RHS
            
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
        far_field = np.sum(p_ff**2, axis=1) # integrates energy of far-field over time.

        # FIX B END: Return the incident power density sum
        return far_field, angles_deg, incident_power_density_sum

    def simulate_scattering(self, T, n_steps=7000, fm=constants.DEFAULT_FM, tau=constants.DEFAULT_TAU, delay=constants.DEFAULT_DELAY):
        # Update the public interface to return all three values
        return self._simulate_scattering(T, n_steps=n_steps, fm=fm, tau=tau, delay=delay)
    
    # This function now needs to be called externally with the incident power density, 
    # but we'll leave it as is for relative scaling, assuming external code handles absolute scaling.
    def calculate_scattering_cross_section(self, far_field):
        H = far_field / np.max(far_field)
        H_dB = 10 * np.log10(H + 1e-12)
        return H_dB