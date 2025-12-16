# run_paper_test.py

import numpy as np
import importlib
import constants 
from simulator import KSpaceAcousticScattering

def run_paper_configuration_short():
    print("--- Running Full Paper Configuration (Short Test) ---")
    print(f"Configuration: N={constants.DEFAULT_N}, r0=2.4m")
    print(f"Time Steps: 100")
    
    # 1. Setup Parameters strictly from Paper
    # N=256, dx=0.02, dt=20e-6 are already in constants.py
    # We explicitly ensure them here just in case.
    constants.DEFAULT_N = 256
    
    # Initialize Simulator
    # (Note: passing dt to satisfy your local file definition if needed)
    sim = KSpaceAcousticScattering(
        N=constants.DEFAULT_N, 
        dx=constants.DEFAULT_DX, 
        dt=constants.DEFAULT_DT
    )
    
    # 2. Create Atmosphere (Bragg r0=2.4m)
    # This is the largest scatterer from Figure 3
    fm = constants.DEFAULT_FM
    r0 = 2.4 
    
    print(f"Creating Bragg Atmosphere (r0={r0}m)...")
    T, window, V_scat = sim.create_bragg_atmosphere(fm, DT=1.0, r0=r0)
    
    # 3. Run Simulation (Short Duration)
    # Note: With default delay (60ms), the pulse will NOT arrive in the first 100 steps (2ms).
    # This primarily tests that the heterogeneous medium doesn't spontaneously explode.
    print("Starting simulation...")
    
    far_field, angles, incident_power = sim.simulate_scattering(
        T, 
        n_steps=100, 
        fm=fm, 
        tau=constants.DEFAULT_TAU, 
        delay=constants.DEFAULT_DELAY
    )
    
    # 4. Analyze Results
    max_val = np.max(np.abs(far_field))
    is_finite = np.isfinite(max_val)
    
    print("\n--- Test Results ---")
    print(f"Finite Output: {is_finite}")
    print(f"Max Scatter Energy: {max_val:.4e}")
    
    if is_finite and max_val < 1e10:
        print("\n[SUCCESS] Simulation is STABLE.")
        print("Note: Energy is expected to be very low/zero because the pulse ")
        print("      has not entered the domain yet (Delay=60ms, Run=2ms).")
    else:
        print("\n[FAILURE] Instability detected (NaN, Inf, or Explosion).")

if __name__ == "__main__":
    # Reload constants to be safe
    constants = importlib.reload(constants)
    run_paper_configuration_short()