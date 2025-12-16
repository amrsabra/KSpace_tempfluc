# profiler.py

import cProfile
import pstats
import constants
from simulator import KSpaceAcousticScattering

def run_profiled_sim(n_steps=100):
    # Full production grid and time step
    N = constants.DEFAULT_N
    dx = constants.DEFAULT_DX
    dt = constants.DEFAULT_DT

    sim = KSpaceAcousticScattering(N=N, dx=dx, dt=dt)

    # Bragg atmosphere
    T, window, V_scat = sim.create_bragg_atmosphere(
        fm=constants.DEFAULT_FM,
        DT=1.0,
        r0=constants.R0,
    )

    # Run simulation (Correctly unpacking 3 values)
    far_field, angles_deg, inc_power = sim.simulate_scattering(
        T,
        n_steps=n_steps,
        fm=constants.DEFAULT_FM,
        tau=constants.DEFAULT_TAU,
        delay=constants.DEFAULT_DELAY,
    )

    return far_field, angles_deg, V_scat

if __name__ == "__main__":
    profile_file = "profile_full_100steps.prof"

    profiler = cProfile.Profile()
    profiler.enable()

    # Run logic
    far_field, _, _ = run_profiled_sim(n_steps=100)

    profiler.disable()
    profiler.dump_stats(profile_file)

    # Print verification
    import numpy as np
    max_energy = np.max(far_field)
    print(f"\n--- Test Results ---")
    print(f"Finite Output: {np.all(np.isfinite(far_field))}")
    print(f"Max Scatter Energy: {max_energy:.4e}")

    if max_energy > 1e10 or np.isnan(max_energy):
        print("\n[FAILURE] Instability detected.")
    else:
        print("\n[SUCCESS] Simulation is stable!")