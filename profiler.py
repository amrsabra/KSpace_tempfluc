# profiler.py

import time
import numpy as np
import constants
from simulator import KSpaceAcousticScattering

def run_profiled_kolmogorov():
    # 1. Setup simulation parameters (matching full grid size and constants)
    N = constants.DEFAULT_N
    dx = constants.DEFAULT_DX
    dt = constants.DEFAULT_DT
    CT2 = 1.5e-7 * constants.T0 ** 2
    r0 = constants.R0
    total_steps = 100
    interval = 10

    print(f"Initializing simulation with N={N} ({N}^3 grid)...")
    sim = KSpaceAcousticScattering(N=N, dx=dx, dt=dt)

    # 2. Create the Kolmogorov atmosphere
    print("Generating Kolmogorov atmosphere...")
    T, window, V_scat = sim.create_kolmogorov_atmosphere(
        CT2=CT2,
        r0=r0,
        seed=42  # Static seed for profiling consistency
    )

    # 3. Setup the simulation (Internal setup usually done in simulate_scattering)
    # We will manually step through the simulation to get the timing intervals
    print(f"Starting profiling for {total_steps} steps...\n")
    
    # Initialization of fields and buffers inside the simulator logic
    sim.fm = constants.DEFAULT_FM
    sim.tau = constants.DEFAULT_TAU
    sim.delay = constants.DEFAULT_DELAY
    sim.ntff.initialize_buffer(total_steps)
    sim.pml.set_dt(sim.dt)

    # Initialize simulation fields (simplified mirror of simulator._simulate_scattering)
    px_s = np.zeros((N, N, N))
    py_s = np.zeros((N, N, N))
    pz_s = np.zeros((N, N, N))
    ux_s = np.zeros((N, N, N))
    uy_s = np.zeros((N, N, N))
    uz_s = np.zeros((N, N, N))
    p_s_total = np.zeros_like(px_s)
    
    rho = constants.RHO0 * constants.T0 / T 
    rho_const_inv = 1.0 / constants.RHO0
    rho_inv = 1.0 / rho
    corr_grad_factor = rho_const_inv - rho_inv
    _, u_i_z_prev = sim.incident.plane_wave(-sim.dt, sim.fm, sim.tau, sim.delay)

    # Timing variables
    overall_start = time.time()
    interval_start = time.time()

    for step in range(1, total_steps + 1):
        # --- CORE SIMULATION STEP (from simulator.py) ---
        t = (step - 1) * sim.dt 
        p_i, u_i_z = sim.incident.plane_wave(t, sim.fm, sim.tau, sim.delay)
        duiz_dt = (u_i_z - u_i_z_prev) / sim.dt
        u_i_z_prev = u_i_z
        source_term = (constants.RHO0 - rho) * duiz_dt
        
        p_s_total[:] = px_s + py_s + pz_s
        dpdx, dpdy, dpdz = sim.kspace_ops.derivatives_xyz(p_s_total)

        rhs_ux = -dpdx * rho_const_inv + (corr_grad_factor * dpdx)
        rhs_uy = -dpdy * rho_const_inv + (corr_grad_factor * dpdy)
        rhs_uz = -dpdz * rho_const_inv + (corr_grad_factor * dpdz) + (source_term / rho)

        sim.pml.update_velocity_component(ux_s, rhs_ux, sim.dt, 'x')
        sim.pml.update_velocity_component(uy_s, rhs_uy, sim.dt, 'y')
        sim.pml.update_velocity_component(uz_s, rhs_uz, sim.dt, 'z')
    
        duxdx = sim.kspace_ops.derivative(ux_s, 'x')
        sim.pml.update_pressure_component(px_s, -constants.RHO0 * constants.C0**2 * duxdx, sim.dt, 'x') 
        duydy = sim.kspace_ops.derivative(uy_s, 'y')
        sim.pml.update_pressure_component(py_s, -constants.RHO0 * constants.C0**2 * duydy, sim.dt, 'y') 
        duzdz = sim.kspace_ops.derivative(uz_s, 'z')
        sim.pml.update_pressure_component(pz_s, -constants.RHO0 * constants.C0**2 * duzdz, sim.dt, 'z')
        
        p_s_total[:] = px_s + py_s + pz_s
        sim.ntff.accumulate(p_s_total, ux_s, uy_s, uz_s, step - 1)
        # --- END CORE STEP ---

        # Logging every 10 steps
        if step % interval == 0:
            interval_end = time.time()
            duration = interval_end - interval_start
            avg_time_per_step = duration / interval
            print(f"Steps {step-9} to {step} done.")
            print(f"  -> Total time for these 10 steps: {duration:.4f} seconds")
            print(f"  -> Average time per step: {avg_time_per_step:.4f} seconds")
            interval_start = time.time() # Reset interval timer

    overall_end = time.time()
    print(f"\nProfiling complete.")
    print(f"Total time for 100 steps: {overall_end - overall_start:.2f} seconds.")

if __name__ == "__main__":
    run_profiled_kolmogorov()