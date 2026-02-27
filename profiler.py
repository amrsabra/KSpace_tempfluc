# profiler.py

import time
import constants
from simulator import KSpaceAcousticScattering
import cProfile  # Required for snakeviz compatibility

def run_profiled_kolmogorov():
    # 1. Setup simulation parameters (matching full grid size and constants)
    N = constants.DEFAULT_N
    dx = constants.DEFAULT_DX
    dt = constants.DEFAULT_DT
    # Boosted 10x to match the new Time-of-Flight calibration in main.py
    CT2 = 1.5e-6 * constants.T0 ** 2 
    total_steps = 100

    print(f"Initializing simulation with N={N} ({N}^3 grid)...")
    sim = KSpaceAcousticScattering(N=N, dx=dx, dt=dt)

    # 2. Create the Kolmogorov atmosphere
    print("Generating Kolmogorov atmosphere...")
    kolm_data_tuple = sim.create_kolmogorov_atmosphere(
        CT2=CT2,
        r0=constants.R0,
        seed=42  # Static seed for profiling consistency
    )

    # 3. Run the simulation through the profiler
    print(f"Starting profiling for {total_steps} steps...\n")
    
    overall_start = time.time()
    
    # We call the exact same method used in the real run to guarantee mathematical sync!
    sim._simulate_scattering(kolm_data_tuple, n_steps=total_steps)
    
    overall_end = time.time()
    
    print(f"\nProfiling complete.")
    print(f"Total time for {total_steps} steps: {overall_end - overall_start:.2f} seconds.")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    run_profiled_kolmogorov()
    profiler.disable()
    
    output_file = "WithWindAdvectionPressureExplosionFixes.prof"
    profiler.dump_stats(output_file)
    print(f"\n[DONE] Profile data saved to {output_file}")