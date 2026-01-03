# profiler.py

import cProfile
import pstats
import numpy as np
import constants
from simulator import KSpaceAcousticScattering

def run_profiled_sim(n_steps=100):
    # 1. Setup Configuration (Paper Specs)
    N = constants.DEFAULT_N  # Should be 256
    dx = constants.DEFAULT_DX # Should be 0.02
    dt = constants.DEFAULT_DT # Should be 20e-6

    print(f"--- Profiling Configuration ---")
    print(f"Grid Size:   {N} x {N} x {N}")
    print(f"Time Steps:  {n_steps}")
    print(f"Grid Spacing: {dx} m")
    print(f"-----------------------------")

    # 2. Initialize Simulator
    # This will allocate the large arrays (approx 1-2 GBs for N=256)
    sim = KSpaceAcousticScattering(N=N, dx=dx, dt=dt)

    # 3. Create Atmosphere (Bragg)
    # We use the standard Bragg atmosphere for the test
    T, window, V_scat = sim.create_bragg_atmosphere(
        fm=constants.DEFAULT_FM,
        DT=1.0,
        r0=constants.R0,
    )

    # 4. Run Simulation
    print(f"Starting simulation loop for {n_steps} steps...")
    
    # Returns: far_field, angles_deg, incident_power_density_sum
    far_field, angles_deg, inc_power = sim.simulate_scattering(
        T,
        n_steps=n_steps,
        fm=constants.DEFAULT_FM,
        tau=constants.DEFAULT_TAU,
        delay=constants.DEFAULT_DELAY,
    )

    return far_field

if __name__ == "__main__":
    profile_file = "profile_full_100steps.prof"

    # Setup Profiler
    profiler = cProfile.Profile()
    
    try:
        profiler.enable()
        
        # --- RUN TEST ---
        far_field = run_profiled_sim(n_steps=100)
        # ----------------
        
        profiler.disable()
        
        # Save Stats
        profiler.dump_stats(profile_file)
        print(f"\n[DONE] Profile saved to: {profile_file}")

        # Verification
        max_energy = np.max(far_field)
        print(f"Max Scatter Energy: {max_energy:.4e}")
        
        if not np.all(np.isfinite(far_field)):
            print("[WARNING] Non-finite values detected in output!")
        
        # --- IMMEDIATE FEEDBACK ---
        print("\n=== Top 15 Functions by Cumulative Time ===")
        # This shows you exactly where the time went (excluding setup)
        ps = pstats.Stats(profiler).sort_stats('cumtime')
        ps.print_stats(15)

    except KeyboardInterrupt:
        print("\n[STOPPED] Profiling interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Profiling failed: {e}")