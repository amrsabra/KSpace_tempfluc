# profile_full_run.py
#
# Profile a single full resolution Bragg run
# N = 256, full NTFF, 360 directions, 100 steps

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

    # Bragg atmosphere at default fm and r0
    T, window, V_scat = sim.create_bragg_atmosphere(
        fm=constants.DEFAULT_FM,
        DT=1.0,
        r0=constants.R0,
    )

    # One full physics run with reduced steps
    far_field, angles_deg = sim.simulate_scattering(
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

    run_profiled_sim(n_steps=100)

    profiler.disable()
    profiler.dump_stats(profile_file)

    # Optional: print a quick summary to the terminal
    stats = pstats.Stats(profile_file)
    stats.sort_stats("cumulative").print_stats(30)

    print(f"\nSaved cProfile data to {profile_file}")
    print("You can inspect it with snakeviz or other tools.")
