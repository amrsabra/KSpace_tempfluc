import sys
import os
import inspect
import numpy as np
import constants

def run_test():
    print("--- 1. Import Debugging ---")
    # Force python to look in current directory first
    sys.path.insert(0, os.getcwd())
    
    try:
        import simulator
        import kspace_operators
    except ImportError as e:
        print(f"CRITICAL: Could not import modules: {e}")
        return

    print(f"Simulator File: {simulator.__file__}")
    print(f"Operators File: {kspace_operators.__file__}")

    # Check if the code in memory actually has the fix
    source = inspect.getsource(simulator.KSpaceAcousticScattering._simulate_scattering)
    
    if "gaussian_filter" in source:
        print("\033[92m[PASS] The loaded 'simulator.py' contains the Gaussian Smoothing fix.\033[0m")
    else:
        print("\033[91m[FAIL] The loaded 'simulator.py' DOES NOT contain the fix!\033[0m")
        print("       Python is loading an old version of the file.")
        print("       Please ensure you are editing the file in the same directory where you run the script.")
        return

    print("\n--- 2. N=256 Stability Test ---")
    # We manually construct the simulation to ensure no other variables interfere
    sim = simulator.KSpaceAcousticScattering(N=256, dx=0.02, dt=20e-6)
    
    print("Creating Bragg Atmosphere...")
    T, _, _ = sim.create_bragg_atmosphere(fm=1000, DT=1.0, r0=2.4)
    
    print("Running 100 steps (High Res)...")
    # Run simulation
    far_field, _, _ = sim.simulate_scattering(T, n_steps=100)
    
    max_energy = np.max(far_field)
    print(f"Max Scatter Energy: {max_energy:.4e}")
    
    if max_energy > 1e10 or np.isnan(max_energy):
        print("\033[91m[FAIL] Simulation Exploded.\033[0m")
    else:
        print("\033[92m[SUCCESS] Simulation is Stable! You can generate your figures.\033[0m")

if __name__ == "__main__":
    run_test()