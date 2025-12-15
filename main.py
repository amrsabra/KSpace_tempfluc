from __future__ import annotations

import argparse
import numpy as np

import constants
from simulator import KSpaceAcousticScattering


# -----------------------------------------------------------------------------
# Bragg atmosphere sweep (Figures 3–5, Table I)
# -----------------------------------------------------------------------------

def run_bragg_sweep(sim: KSpaceAcousticScattering,
                     n_steps: int,
                     fm: float,
                     tau_short: float,
                     r0_values: list[float]) -> dict[str, np.ndarray]:
    """Run the Bragg atmosphere sweep for the 5 radii in Table I.

    For each outer radius r0 in ``r0_values`` this function:
    - builds the Bragg atmosphere (Eq. 19 with window Eq. 18),
    - runs the k-space scattered-field simulation (Eqs. 9–11),
    - evaluates the far-field via the NTFF transform (Eq. 17).

    The NPZ output is then used by ``visualization.py`` to reproduce
    Figures 3–5 and Table I (H(θ), W, H(180°), Q, V_scat).
    """
    bragg_far_field_energy_list: list[np.ndarray] = []
    bragg_p_ff_list: list[np.ndarray] = []
    bragg_V_scat_list: list[float] = []
    angles_deg: np.ndarray | None = None

    for r0 in r0_values:
        print(f"\n=== Bragg atmosphere: r0 = {r0:.3f} m ===")

        # Build Bragg atmosphere for this scattering radius (Eq. 19)
        T, window, V_scat = sim.create_bragg_atmosphere(
            fm=fm,
            DT=1.0,   # ΔT = 1 K as in the paper
            r0=r0,
        )
        bragg_V_scat_list.append(V_scat)

        # Run scattered-field simulation
        far_field_energy, angles_deg = sim.simulate_scattering(
            T,
            n_steps=n_steps,
            fm=fm,
            tau=tau_short,                 # short pulse r = 1 ms (Sec. III A)
            delay=constants.DEFAULT_DELAY,
        )

        # NTFF time-series p_ff(θ, t)
        p_ff = sim.ntff.compute_far_field()

        bragg_far_field_energy_list.append(far_field_energy)
        bragg_p_ff_list.append(p_ff)

    bragg_far_field_energy = np.stack(bragg_far_field_energy_list, axis=0)
    bragg_p_ff = np.stack(bragg_p_ff_list, axis=0)
    bragg_V_scat = np.array(bragg_V_scat_list, dtype=float)

    return {
        "bragg_r0": np.array(r0_values, dtype=float),
        "bragg_V_scat": bragg_V_scat,
        "bragg_far_field_energy": bragg_far_field_energy,
        "bragg_p_ff": bragg_p_ff,
        "angles_deg": np.array(angles_deg, dtype=float),
    }


# -----------------------------------------------------------------------------
# Kolmogorov ensemble (Figures 6–7, Table II)
# -----------------------------------------------------------------------------

def run_kolmogorov_ensemble(sim: KSpaceAcousticScattering,
                             n_steps: int,
                             fm_list: list[float],
                             n_realizations: int,
                             CT2: float) -> dict[str, np.ndarray]:
    """Run Kolmogorov atmosphere ensemble for Fig. 6–7 and Table II.

    For each realization (seed) we:
    - generate a Kolmogorov atmosphere with CT^2 given by Eq. (22),
    - run the scattered-field model for fm=1 kHz and 1.2 kHz,
    - accumulate far-field p_ff and energy.

    The resulting dataset is used to compute:
    - Fig. 6: example temperature slice (T - T0),
    - Fig. 7: mean H(θ) vs analytical curve (Eq. 23),
    - Table II: backscattered phase and equivalent delay.
    """
    seeds = np.arange(n_realizations, dtype=int)
    first_run = True

    kolm_p_ff: np.ndarray | None = None
    kolm_far_field_energy: np.ndarray | None = None
    angles_deg: np.ndarray | None = None
    kolm_V_scat: float | None = None
    kolm_T_example: np.ndarray | None = None

    for i, seed in enumerate(seeds):
        print(f"\n=== Kolmogorov atmosphere: seed = {seed} ===")

        # Generate one Kolmogorov atmosphere instance (Fig. 6, Eq. 22)
        T, window, V_scat = sim.create_kolmogorov_atmosphere(
            CT2=CT2,
            r0=constants.R0,
            seed=int(seed),
        )
        if kolm_V_scat is None:
            kolm_V_scat = float(V_scat)

        # Save a single example of T - T0 for plotting (Fig. 6)
        if kolm_T_example is None:
            kolm_T_example = T - constants.T0

        for j, fm in enumerate(fm_list):
            print(f"  -> fm = {fm:.1f} Hz")

            far_field_energy, angles_deg = sim.simulate_scattering(
                T,
                n_steps=n_steps,
                fm=fm,
                tau=constants.DEFAULT_TAU,      # long pulse r = 10 ms
                delay=constants.DEFAULT_DELAY,
            )

            p_ff = sim.ntff.compute_far_field()

            if first_run:
                n_dirs, n_time = p_ff.shape
                n_fm = len(fm_list)
                kolm_p_ff = np.zeros(
                    (n_realizations, n_fm, n_dirs, n_time),
                    dtype=np.complex128,
                )
                kolm_far_field_energy = np.zeros(
                    (n_realizations, n_fm, n_dirs),
                    dtype=np.float64,
                )
                first_run = False

            assert kolm_p_ff is not None
            assert kolm_far_field_energy is not None

            kolm_p_ff[i, j, :, :] = p_ff
            kolm_far_field_energy[i, j, :] = far_field_energy

    assert kolm_p_ff is not None
    assert kolm_far_field_energy is not None
    assert angles_deg is not None
    assert kolm_T_example is not None
    assert kolm_V_scat is not None

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


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run k-space acoustic scattering simulations for Bragg and "
            "Kolmogorov atmospheres (Hargreaves et al. 2014, Figs. 3–7, "
            "Tables I–II)."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["all", "bragg", "kolm", "bragg_single", "kolm_single"],
        default="all",
    )
    parser.add_argument("--output", type=str, default="scattering_results_fig3to7.npz")
    parser.add_argument("--n_steps", type=int, default=7000)
    parser.add_argument("--r0", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed index for mode=kolm_single")
    args = parser.parse_args()

    # Simulation parameters from Sec. III (N, dx, dt)
    sim = KSpaceAcousticScattering(
        N=constants.DEFAULT_N,
        dx=constants.DEFAULT_DX,
        dt=constants.DEFAULT_DT,
    )

    # Time axis for output (common to all runs)
    t = np.arange(args.n_steps, dtype=float) * constants.DEFAULT_DT

    # Bragg parameters (Sec. III A, Eq. 19, Table I)
    tau_short = 1e-3                 # r = 1 ms for short pulse
    bragg_r0_values = [0.15, 0.3, 0.6, 1.2, 2.4]
    bragg_fm = 1000.0               # 1 kHz

    # Kolmogorov parameters (Sec. III B, Eq. 22, Table II)
    CT2 = 1.5e-7 * constants.T0**2  # C_T^2 = 1.5 × 10^{-7} T0^2
    fm_list = [1000.0, 1200.0]
    n_realizations = 8

    out: dict[str, np.ndarray | float | int | str] = {
        "dx": float(constants.DEFAULT_DX),
        "dt": float(constants.DEFAULT_DT),
        "N": int(constants.DEFAULT_N),
        "T0": float(constants.T0),
        "rho0": float(constants.RHO0),
        "c0": float(constants.C0),
        "PML_DEPTH": int(constants.PML_DEPTH),
        "t": t,
        "mode": args.mode,
    }

    # ------------------------------------------------------------------
    # Bragg: full sweep (Figs. 3–5, Table I)
    # ------------------------------------------------------------------
    if args.mode in ("all", "bragg"):
        bragg_data = run_bragg_sweep(
            sim,
            args.n_steps,
            float(bragg_fm),
            tau_short,
            bragg_r0_values,
        )
        out.update(bragg_data)

    # ------------------------------------------------------------------
    # Kolmogorov: full ensemble (Figs. 6–7, Table II)
    # ------------------------------------------------------------------
    if args.mode in ("all", "kolm"):
        kolm_data = run_kolmogorov_ensemble(
            sim,
            args.n_steps,
            fm_list,
            n_realizations,
            CT2,
        )
        # avoid duplicating angles_deg key if Bragg already set it
        if "angles_deg" in out and "angles_deg" in kolm_data:
            kolm_data.pop("angles_deg", None)
        out.update(kolm_data)

    # ------------------------------------------------------------------
    # Bragg: single radius (for job arrays)
    # ------------------------------------------------------------------
    if args.mode == "bragg_single":
        if args.r0 is None:
            raise SystemExit("mode=bragg_single requires --r0")

        r0 = float(args.r0)
        print(f"\n=== Single Bragg run, r0 = {r0:.3f} m ===")

        T, window, V_scat = sim.create_bragg_atmosphere(
            fm=bragg_fm,
            DT=1.0,
            r0=r0,
        )
        far_field_energy, angles_deg = sim.simulate_scattering(
            T,
            n_steps=args.n_steps,
            fm=bragg_fm,
            tau=tau_short,
            delay=constants.DEFAULT_DELAY,
        )
        p_ff = sim.ntff.compute_far_field()

        out.update({
            "mode": "bragg_single",
            "r0": r0,
            "V_scat": float(V_scat),
            "far_field_energy": far_field_energy,
            "p_ff": p_ff,
            "angles_deg": np.array(angles_deg, dtype=float),
        })

    # ------------------------------------------------------------------
    # Kolmogorov: single realization (for job arrays)
    # ------------------------------------------------------------------
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

        kolm_p_ff_list: list[np.ndarray] = []
        kolm_energy_list: list[np.ndarray] = []
        angles_deg: np.ndarray | None = None

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

        assert angles_deg is not None

        out.update({
            "mode": "kolm_single",
            "seed": seed,
            "kolm_V_scat": float(V_scat),
            # shape: (2, n_dirs, n_time) for fm = [1000, 1200]
            "kolm_p_ff": np.array(kolm_p_ff_list),
            # shape: (2, n_dirs)
            "kolm_far_field_energy": np.array(kolm_energy_list),
            "angles_deg": np.array(angles_deg, dtype=float),
            "kolm_T_example": T - constants.T0,
        })

    print(f"\nSaving results to {args.output} ...")
    np.savez(args.output, **out)
    print("Done.")


if __name__ == "__main__":
    main()
