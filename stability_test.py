# stability_test.py
import numpy as np
import constants
from simulator import KSpaceAcousticScattering

def run_test(use_original_values=False):
    if use_original_values:
        # These are the values that likely caused the HPC crash
        TEST_N = 256
        TEST_DT = 20e-6
        print("\n--- WARNING: RUNNING ORIGINAL VALUES (Memory Intensive) ---")
        print("Note: This requires ~20GB of RAM. If you have less, this will crash your PC.")
    else:
        # Safe values for local debugging
        TEST_N = 128
        TEST_DT = 10e-6
        print("\n--- Running Safe Local Test ---")

    N_STEPS = 500
    print(f"Grid: {TEST_N}^3, DT: {TEST_DT}s, Steps: {N_STEPS}")

    # Initialize simulator
    sim = KSpaceAcousticScattering(N=TEST_N, dx=constants.DEFAULT_DX, dt=TEST_DT)
    
    # Create Kolmogorov atmosphere
    CT2 = 1.5e-7 * constants.T0 ** 2
    kolm_data = sim.create_kolmogorov_atmosphere(CT2=CT2, seed=42)
    
    try:
        sim._simulate_scattering(kolm_data, n_steps=N_STEPS)
        print("[SUCCESS] Simulation stayed stable.")
    except FloatingPointError:
        print("[FAILED] Simulation exploded as expected.")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

if __name__ == "__main__":
    # 1. Run the safe one first
    run_test(use_original_values=False)
    
    # 2. Uncomment below to see the "Original" fail (Check your RAM first!)
    # run_test(use_original_values=True)