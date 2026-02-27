# main.py

from __future__ import annotations

import argparse
import numpy as np

import constants
from simulator import KSpaceAcousticScattering


def run_bragg_sweep(sim, n_steps, fm, tau_short, r0_values):
    # Run the Bragg atmosphere sweep for the 5 different radii (Fig. 3, 4, 5, Table I).
    bragg_far_field_energy_list = []
    bragg_p_ff_list = []
    bragg_V_scat_list = []
    angles_deg = None
    incident_power_density_list = [] 
    sensor_fwd_list = []
    sensor_bwd_list = []

    for r0 in r0_values:
        print(f"\n=== Bragg atmosphere: r0 = {r0:.3f} m ===")

        # Create Bragg atmosphere for specific radius
        T, window, V_scat = sim.create_bragg_atmosphere(
            fm=fm,
            DT=1.0,
            r0=r0,
        )
        bragg_V_scat_list.append(V_scat)

        # Capture the three return values (far_field_energy, angles_deg, incident_power_density_sum)
        far_field_energy, angles_deg, incident_power_density_sum, sensor_fwd, sensor_bwd = sim.simulate_scattering(
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
        incident_power_density_list.append(incident_power_density_sum)
        sensor_fwd_list.append(sensor_fwd)
        sensor_bwd_list.append(sensor_bwd)

    bragg_far_field_energy = np.stack(bragg_far_field_energy_list, axis=0)
    bragg_p_ff = np.stack(bragg_p_ff_list, axis=0)
    bragg_V_scat = np.array(bragg_V_scat_list)

    return {
        "bragg_r0": np.array(r0_values, dtype=float),
        "bragg_V_scat": bragg_V_scat,
        "bragg_far_field_energy": bragg_far_field_energy,
        "bragg_p_ff": bragg_p_ff,
        "angles_deg": np.array(angles_deg, dtype=float),
        "bragg_incident_power_density": np.array(incident_power_density_list[0], dtype=float),
        "bragg_sensor_fwd_raw": np.array(sensor_fwd_list),
        "bragg_sensor_bwd_raw": np.array(sensor_bwd_list),
    }


def run_kolmogorov_ensemble(sim, n_steps, fm_list, n_realizations, CT2):
    # Run Kolmogorov atmosphere ensemble (Figs 6-7, Table II).
    seeds = np.arange(n_realizations, dtype=int)
    first_run = True

    kolm_p_ff = None
    kolm_far_field_energy = None
    angles_deg = None
    kolm_V_scat = None
    kolm_T_example = None
    kolm_incident_power_density = None 
    kolm_sensor_fwd = None
    kolm_sensor_bwd = None

    for i, seed in enumerate(seeds):
        print(f"\n=== Kolmogorov atmosphere: seed = {seed} ===")

        kolm_data_tuple = sim.create_kolmogorov_atmosphere(CT2=CT2, r0=constants.R0, seed=seed)
        T, window, V_scat, _, _ = kolm_data_tuple

        if kolm_V_scat is None: # volume scattering
            kolm_V_scat = V_scat

        if kolm_T_example is None: # temp fluc 
            kolm_T_example = T - constants.T0

        for j, fm in enumerate(fm_list): # loop to show how different sound frequencies interact with same atmosphere (1000Hz and 1200Hz)
            print(f"  -> fm = {fm:.1f} Hz")
            far_field_energy, angles_deg, incident_power_density_sum, sensor_fwd, sensor_bwd = sim.simulate_scattering(
                kolm_data_tuple,
                n_steps=n_steps,
                fm=fm,
                tau=constants.DEFAULT_TAU,
                delay=constants.DEFAULT_DELAY,
            )

            p_ff = sim.ntff.compute_far_field()

            if first_run: #initialise needed variables only the first time, and reuse the second time
                n_dirs, n_time = p_ff.shape
                n_fm = len(fm_list)
                kolm_p_ff = np.zeros((n_realizations, n_fm, n_dirs, n_time), dtype=np.complex128)
                kolm_far_field_energy = np.zeros((n_realizations, n_fm, n_dirs), dtype=np.float64)
                kolm_incident_power_density = np.zeros(n_fm, dtype=np.float64)
                kolm_sensor_fwd = np.zeros((n_realizations, n_fm, n_steps), dtype=np.float64)
                kolm_sensor_bwd = np.zeros((n_realizations, n_fm, n_steps), dtype=np.float64)
                first_run = False

            '''
            i,j are indices and : tell the computer exactly which slot to put data in
            i (realization index) which shows which realisation we are in
            j (frequency index) showing which frequency was used
            :, : tells NumPy to take the 2D result of p_ff (360 directions by 7000 steps) and store it with specific i and j coordinates
            '''
            kolm_p_ff[i, j, :, :] = p_ff 
            kolm_far_field_energy[i, j, :] = far_field_energy
            kolm_sensor_fwd[i, j, :] = sensor_fwd
            kolm_sensor_bwd[i, j, :] = sensor_bwd
            
            # Store incident power density (constant across realizations, but depends on fm)
            if i == 0:
                 kolm_incident_power_density[j] = incident_power_density_sum


    return {
        "kolm_seeds": seeds,
        "kolm_CT2": float(CT2),
        "kolm_fm_list": np.array(fm_list, dtype=float),
        "kolm_V_scat": float(kolm_V_scat),
        "kolm_p_ff": kolm_p_ff,
        "kolm_far_field_energy": kolm_far_field_energy,
        "kolm_T_example": kolm_T_example,
        "angles_deg": np.array(angles_deg, dtype=float),
        "kolm_incident_power_density": kolm_incident_power_density,
        "kolm_sensor_fwd_raw": kolm_sensor_fwd,
        "kolm_sensor_bwd_raw": kolm_sensor_bwd,
    }

def main():
    parser = argparse.ArgumentParser(description="Run k-space acoustic scattering sims and save results to NPZ.")
    parser.add_argument(
        "--mode",
        choices=["all", "bragg", "kolm", "bragg_single", "kolm_single"],
        default="all",
    )
    parser.add_argument("--output", type=str, default="scattering_results.npz")
    parser.add_argument("--n_steps", type=int, default=7000)
    parser.add_argument("--r0", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    sim = KSpaceAcousticScattering(
        N=constants.DEFAULT_N,
        dx=constants.DEFAULT_DX,
        dt=constants.DEFAULT_DT,
    )

    t = np.arange(args.n_steps) * constants.DEFAULT_DT
    tau_short = 1e-3
    bragg_r0_values = [0.15, 0.3, 0.6, 1.2, 2.4]
    CT2 = 1.5e-6 * constants.T0 ** 2 #From the paper right under EQN 22.
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

    # Bragg: full sweep or combined run (old)
    if args.mode in ("all", "bragg"):
        bragg_data = run_bragg_sweep(
            sim,
            args.n_steps,
            float(constants.DEFAULT_FM),
            tau_short,
            bragg_r0_values,
        )
        out.update(bragg_data)

    # Kolmogorov: full ensemble in one job (old)
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

    # Bragg: single radius (for job array) (new)
    if args.mode == "bragg_single":
        if args.r0 is None:
            raise SystemExit("mode=bragg_single requires --r0")

        print(f"\n=== Single Bragg run, r0 = {args.r0:.3f} m ===")

        T, window, V_scat = sim.create_bragg_atmosphere(
            fm=float(constants.DEFAULT_FM),
            DT=1.0,
            r0=float(args.r0),
        )

        far_field_energy, angles_deg, incident_power_density_sum, sensor_fwd, sensor_bwd = sim.simulate_scattering(
            T, # Pass T directly for Bragg
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
            # FIX: Include incident power density
            "incident_power_density": float(incident_power_density_sum),
            "sensor_fwd_raw": sensor_fwd,
            "sensor_bwd_raw": sensor_bwd,
        })

    # Kolmogorov: single realization (for job array) (new)
    if args.mode == "kolm_single":
        if args.seed is None:
            raise SystemExit("mode=kolm_single requires --seed")

        seed = int(args.seed)
        print(f"\n=== Kolmogorov single run, seed = {seed} ===")

        kolm_data_tuple = sim.create_kolmogorov_atmosphere(CT2=CT2, r0=constants.R0, seed=seed)
        T, window, V_scat, _, _ = kolm_data_tuple

        kolm_p_ff_list = []
        kolm_energy_list = []
        angles_deg = None
        kolm_incident_power_list = [] # List to store incident power for each fm
        sensor_fwd_list = []
        sensor_bwd_list = []

        for fm in fm_list:
            print(f"  -> fm = {fm:.1f} Hz")
            far_field_energy, angles_deg, incident_power_density_sum, sensor_fwd, sensor_bwd = sim.simulate_scattering(
                kolm_data_tuple,
                n_steps=args.n_steps,
                fm=fm,
                tau=constants.DEFAULT_TAU,
                delay=constants.DEFAULT_DELAY,
            )
            p_ff = sim.ntff.compute_far_field()

            kolm_p_ff_list.append(p_ff)
            kolm_energy_list.append(far_field_energy)
            kolm_incident_power_list.append(incident_power_density_sum)
            sensor_fwd_list.append(sensor_fwd)
            sensor_bwd_list.append(sensor_bwd)

        out.update({
            "mode": "kolm_single",
            "seed": seed,
            "kolm_V_scat": float(V_scat),
            "kolm_window": window,
            # shape: (2, n_dirs, n_time) for fm = [1000, 1200]
            "kolm_p_ff": np.array(kolm_p_ff_list),
            # shape: (2, n_dirs)
            "kolm_far_field_energy": np.array(kolm_energy_list),
            "angles_deg": np.array(angles_deg, dtype=float),
            "kolm_incident_power_density": np.array(kolm_incident_power_list, dtype=float),
            "kolm_T_example": T - constants.T0,
            "sensor_fwd_raw": np.array(sensor_fwd_list),
            "sensor_bwd_raw": np.array(sensor_bwd_list),
        })

    print(f"\nSaving results to {args.output} ...")
    np.savez(args.output, **out)
    print("Done.")


if __name__ == "__main__":
    main()