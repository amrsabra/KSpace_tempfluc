# K Space Acoustic Scattering (Python Implementation)

This project is a Python implementation of the scattering model from the paper "Acoustic scattering using a k space method".  
The code reproduces Bragg and Kolmogorov atmospheres and their scattering patterns, and is used for Figures 3, 6, and 7.

---

## Overview of Equations and Where They Appear

- **Eq 1 and Eq 2**  
  Incident pressure and particle velocity for the plane wave. Implemented in `incident_wave.py` and used from `simulator.py` for the source term.

- **Eq 5 and Eq 6**  
  Decomposition into incident plus scattered fields. Not coded as explicit formulas, but the whole solver is written in terms of the scattered fields only. The incident field stays analytical, and the numerical update only evolves the scattered part in `simulator.py`.

- **Eq 9**  
  Momentum equation for the scattered velocity including the source term with density contrast. Implemented inside the velocity update loop in `_simulate_scattering` in `simulator.py`.

- **Eq 10**  
  Continuity equation for the scattered pressure. Implemented inside the pressure update loop in `_simulate_scattering` in `simulator.py`.

- **Eq 13**  
  Exponential time stepping inside the PML. Implemented in `pml.py` in the `update_velocity_component` and `update_pressure_component` methods, and used from `simulator.py`.

- **Eq 17**  
  Near to far field integral. Implemented in `ntff.py` (geometry, retardation times, surface integral and time derivative) and called from `simulator.py`.

- **Eq 18**  
  Bragg type periodic temperature structure inside a spherical window. Implemented in `create_bragg_atmosphere` in `atmospheres.py`, used from `visualization.py` for Figure 3.

- **Eq 22**  
  Kolmogorov spectrum in k space. Implemented in `create_kolmogorov_atmosphere` in `atmospheres.py`, used from `visualization.py` for Figures 6 and 7.

- **Eq 23**  
  Analytical scattering cross section for the Kolmogorov atmosphere. Implemented directly in `generate_figure_7` in `visualization.py` when computing the red analytical curve.

The k space spatial derivatives used in Eq 9 and Eq 10 are implemented via FFT in `kspace_operators.py`.

---

## File by File Explanation

### `constants.py`

**Purpose**  
- Stores all physical constants and default simulation parameters  
  T0, RHO0, C0, grid size, time step, PML settings, and default pulse parameters

**Equations**  
- No single equation from the paper is implemented here  
- Values are used in Eq 1, Eq 2, Eq 9, Eq 10, Eq 18, Eq 22, Eq 23 in other files

**Figures**  
- Supports all figures indirectly  
  Figure 3, Figure 6, Figure 7

---

### `atmospheres.py`

**Purpose**  
- Builds the background temperature fields inside a Tukey spherical window  
- Two models: Bragg atmosphere and Kolmogorov turbulence atmosphere

**Equations**  
- Tukey window concept that defines the scattering volume  
  Used when computing `V_scat` by summing the window  
- **Eq 18**  
  Implemented in `create_bragg_atmosphere`  
- **Eq 22**  
  Implemented in `create_kolmogorov_atmosphere`  

---

### `incident_wave.py`

**Purpose**  
- Defines the analytical vertical plane wave travelling in plus z direction  
- Provides incident pressure and incident z velocity as closed forms

**Equations**  
- **Eq 1**  
  Implemented as the Gaussian modulated cosine for pressure `p_i`  
- **Eq 2**  
  Implemented as `u_i_z = p_i / (rho0 * c0)` for the incident velocity  
- The time delay `tau = t minus z over c0` builds the correct propagation

---

### `kspace_operators.py`

**Purpose**  
- Implements k space derivative operators with FFT and a sinc term  
- Handles spatial derivatives in x, y, z for pressure and velocity

**Equations**  
- Supports the gradient and divergence terms that appear in Eq 9 and Eq 10  
- The sinc term corresponds to the k space correction that makes the scheme accurate at larger time steps

---

### `pml.py`

**Purpose**  
- Implements the absorbing boundary layer at the edges of the domain  
- Stores alpha profiles in x, y, z and applies exponential time stepping

**Equations**  
- **Eq 13**  
  Implemented in `update_velocity_component` and `update_pressure_component` 

**Usage**  
- `simulator.py` calls `update_velocity_component` for ux, uy, uz  
- `simulator.py` calls `update_pressure_component` for px, py, pz

---

### `ntff.py`

**Purpose**  
- Implements the near to far field transform on a six face box around the scattering region

---

### `simulator.py`

**Purpose**  
- Main k space scattering solver  
- Ties together the grid, atmosphere, incident wave, PML, k space derivatives, and NTFF

**Usage**  
- `visualization.py` calls  
  - `create_bragg_atmosphere` or `create_kolmogorov_atmosphere` through the `atmosphere` object  
  - `simulate_scattering` to run Eq 9 and Eq 10 with PML and NTFF  
  - `calculate_scattering_cross_section` to get H_dB

**Figures**  
- **Figure 3**  
  Bragg atmosphere plus scattering solver and NTFF  
- **Figure 7**  
  Kolmogorov atmosphere plus scattering solver and NTFF for multiple realizations

---

### `main.py`

**Purpose**  
- Simple driver to run all figures in one go

---

## HPC Execution (SLURM)

To run the code on a university HPC cluster using SLURM, create a submission script (e.g., `run_kspace.sh`) with the following content: (I put the maximum amount of resources i could to see if it would be enough to model. normally, you could use 8 cpu cores, and 32-64G of memory)

```bash
#!/bin/bash -l
#SBATCH --job-name=kspace_3_7_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH -p amd_student
#SBATCH --output=logs/kspace_%j.out
#SBATCH --error=logs/kspace_%j.err

module purge
module load conda/python3

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

/home/aams1e22/conda-envs/kspace/bin/python -u main.py
```

**Note:** Make sure to:
1. Create the `logs/` directory before submitting: `mkdir -p logs`
2. Update the conda environment path (`/home/user/conda-envs/kspace/bin/python`) to match your HPC setup
3. Submit the job with: `sbatch run_kspace.sh`

