from __future__ import annotations

import argparse
import numpy as np

import constants
from simulator import KSpaceAcousticScattering


def run_bragg_sweep(sim, n_steps, fm, tau_short, r0_values):
    """Run the Bragg atmosphere sweep for the 5 different radii (Fig. 3, 4, 5, Table I).
    Returns a dict of arrays to be saved into the npz.
    """
    bragg_far_field_energy_list = []
    bragg_p_ff_list = []
    bragg_V_scat_list = []
    angles_deg = None

    for r0 in r0_values:
        print(f"\n=== Bragg atmosphere: r0 = {r0:.3f} m ===")

        # Create Bragg atmosphere for this radius
        T, window, V_scat = sim.create_bragg_atmosphere(
            fm=fm,
            DT=1.0,
            r0=r0,
        )
        bragg_V_scat_list.append(V_scat)

        # Run k-space scattering simulation
        far_field_energy, angles_deg = sim.simulate_scattering(
            T,
            n_steps=n_steps,
            fm=fm,
            tau=tau_short,
            delay=constants.DEFAULT_DELAY,
        )

        # Recover full far-field time series p_ff from NTFF buffer
        p_ff = sim.ntff.compute_far_field()

        bragg_far_field_energy_list.append(far_field_energy)
        bragg_p_ff_list.append(p_ff)

    bragg_far_field_energy = np.stack(bragg_far_field_energy_list, axis=0)
    bragg_p_ff = np.stack(bragg_p_ff_list, axis=0)
    bragg_V_scat = np.array(bragg_V_scat_list)

    return {
        "bragg_r0": np.array(r0_values, dtype=float),
        "bragg_V_scat": bragg_V_scat,
        "bragg_far_field_energy": bragg_far_field_energy,
        "bragg_p_ff": bragg_p_ff,
        "angles_deg": np.array(angles_deg, dtype=float),
    }


def run_kolmogorov_ensemble(sim, n_steps, fm_list, n_realizations, CT2):
    """Run Kolmogorov atmosphere ensemble (Figs 6–7, Table II).
    Uses the same temperature realization for both fm=1000 and fm=1200.
    """
    seeds = np.arange(n_realizations, dtype=int)
    first_run = True

    kolm_p_ff = None
    kolm_far_field_energy = None
    angles_deg = None
    kolm_V_scat = None
    kolm_T_example = None

    for i, seed in enumerate(seeds):
        print(f"\n=== Kolmogorov atmosphere: seed = {seed} ===")

        T, window, V_scat = sim.create_kolmogorov_atmosphere(
            CT2=CT2,
            r0=constants.R0,
            seed=int(seed),
        )
        if kolm_V_scat is None:
            kolm_V_scat = V_scat

        if kolm_T_example is None:
            kolm_T_example = T - constants.T0

        for j, fm in enumerate(fm_list):
            print(f"  -> fm = {fm:.1f} Hz")

            far_field_energy, angles_deg = sim.simulate_scattering(
                T,
                n_steps=n_steps,
                fm=fm,
                tau=constants.DEFAULT_TAU,
                delay=constants.DEFAULT_DELAY,
            )

            p_ff = sim.ntff.compute_far_field()

            if first_run:
                n_dirs, n_time = p_ff.shape
                n_fm = len(fm_list)
                kolm_p_ff = np.zeros((n_realizations, n_fm, n_dirs, n_time), dtype=np.complex128)
                kolm_far_field_energy = np.zeros((n_realizations, n_fm, n_dirs), dtype=np.float64)
                first_run = False

            kolm_p_ff[i, j, :, :] = p_ff
            kolm_far_field_energy[i, j, :] = far_field_energy

    return {
        "kolm_seeds": seeds,
        "kolm_CT2": float(CT2),
        "kolm_fm_list": np.array(fm_list, dtype=float),
        "kolm_V_scat": float(kolm_V_scat),
        "kolm_p_ff": kolm_p_ff,
        "kolm_far_field_energy": kolm_far_field_energy,
        "kolm_T_example": kolm_T_example,
        "angles_deg": np.array(angles_deg, dtype=float),
    }


def main():
    parser = argparse.ArgumentParser(description="Run k-space acoustic scattering sims and save results to NPZ.")
    parser.add_argument(
        "--mode",
        choices=["all", "bragg", "kolm", "bragg_single", "kolm_single"],
        default="all",
    )
    parser.add_argument("--output", type=str, default="scattering_results_fig3to7.npz")
    parser.add_argument("--n_steps", type=int, default=7000)
    parser.add_argument("--r0", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Seed index for mode=kolm_single")
    args = parser.parse_args()

    sim = KSpaceAcousticScattering(
        N=constants.DEFAULT_N,
        dx=constants.DEFAULT_DX,
        dt=constants.DEFAULT_DT,
    )

    t = np.arange(args.n_steps) * constants.DEFAULT_DT
    tau_short = 1e-3
    bragg_r0_values = [0.15, 0.3, 0.6, 1.2, 2.4]
    CT2 = 1.5e-7 * constants.T0 ** 2
    fm_list = [1000.0, 1200.0]
    n_realizations = 8

    out = {
        "dx": float(constants.DEFAULT_DX),
        "dt": float(constants.DEFAULT_DT),
        "N": int(constants.DEFAULT_N),
        "T0": float(constants.T0),
        "rho0": float(constants.RHO0),
        "c0": float(constants.C0),
        "PML_DEPTH": int(constants.PML_DEPTH),
        "t": t,
    }

    # Bragg: full sweep or combined run
    if args.mode in ("all", "bragg"):
        bragg_data = run_bragg_sweep(
            sim,
            args.n_steps,
            float(constants.DEFAULT_FM),
            tau_short,
            bragg_r0_values,
        )
        out.update(bragg_data)

    # Kolmogorov: full ensemble in one job (old behaviour)
    if args.mode in ("all", "kolm"):
        kolm_data = run_kolmogorov_ensemble(
            sim,
            args.n_steps,
            fm_list,
            n_realizations,
            CT2,
        )
        if "angles_deg" in out and "angles_deg" in kolm_data:
            kolm_data.pop("angles_deg", None)
        out.update(kolm_data)

    # Bragg: single radius (for job array)
    if args.mode == "bragg_single":
        if args.r0 is None:
            raise SystemExit("mode=bragg_single requires --r0")

        print(f"\n=== Single Bragg run, r0 = {args.r0:.3f} m ===")

        T, window, V_scat = sim.create_bragg_atmosphere(
            fm=float(constants.DEFAULT_FM),
            DT=1.0,
            r0=float(args.r0),
        )
        far_field_energy, angles_deg = sim.simulate_scattering(
            T,
            n_steps=args.n_steps,
            fm=float(constants.DEFAULT_FM),
            tau=tau_short,
            delay=constants.DEFAULT_DELAY,
        )
        p_ff = sim.ntff.compute_far_field()

        out.update({
            "mode": "bragg_single",
            "r0": float(args.r0),
            "V_scat": float(V_scat),
            "far_field_energy": far_field_energy,
            "p_ff": p_ff,
            "angles_deg": np.array(angles_deg, dtype=float),
        })

    # Kolmogorov: single realization (for job array)
    if args.mode == "kolm_single":
        if args.seed is None:
            raise SystemExit("mode=kolm_single requires --seed")

        seed = int(args.seed)
        print(f"\n=== Kolmogorov single run, seed = {seed} ===")

        T, window, V_scat = sim.create_kolmogorov_atmosphere(
            CT2=CT2,
            r0=constants.R0,
            seed=seed,
        )

        kolm_p_ff_list = []
        kolm_energy_list = []
        angles_deg = None

        for fm in fm_list:
            print(f"  -> fm = {fm:.1f} Hz")
            far_field_energy, angles_deg = sim.simulate_scattering(
                T,
                n_steps=args.n_steps,
                fm=fm,
                tau=constants.DEFAULT_TAU,
                delay=constants.DEFAULT_DELAY,
            )
            p_ff = sim.ntff.compute_far_field()

            kolm_p_ff_list.append(p_ff)
            kolm_energy_list.append(far_field_energy)

        out.update({
            "mode": "kolm_single",
            "seed": seed,
            "kolm_V_scat": float(V_scat),
            # shape: (2, n_dirs, n_time) for fm = [1000, 1200]
            "kolm_p_ff": np.array(kolm_p_ff_list),
            # shape: (2, n_dirs)
            "kolm_far_field_energy": np.array(kolm_energy_list),
            "angles_deg": np.array(angles_deg, dtype=float),
            # store T - T0 for this seed (optional; you can keep only seed 0 later if you wish)
            "kolm_T_example": T - constants.T0,
        })

    print(f"\nSaving results to {args.output} ...")
    np.savez(args.output, **out)
    print("Done.")


if __name__ == "__main__":
    main()
