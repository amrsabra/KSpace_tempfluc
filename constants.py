#constants.py
# This file contains the physical constants and simulation parameters.

# Uniform part of the atmosphere.
T0 = 288.0      # Temperature (K)
RHO0 = 1.22     # Density (kg/m^3)
C0 = 340.0      # Speed of sound (m/s)

# Default simulation parameters
DEFAULT_N = 256          # Number of grid points (grid is a cube, Nx = Ny = Nz)
DEFAULT_DX = 0.02        # Grid spacing (m) -> Physical Grid Size = 5.12m
DEFAULT_DT = 20e-6       # Time step (s) -> every 1000 steps is 0.02s of simulation time 

# PML parameters
PML_DEPTH = 8            # PML depth in grid cells
PML_ABSORPTION = 16.0    # PML absorption coefficient (Nepers)

# Kolmogorov spectrum parameters
KOLMOGOROV_CONSTANT = 0.033

# Default pulse parameters
DEFAULT_FM = 1000        # Modulation frequency (Hz)
DEFAULT_TAU = 10e-3      # Pulse duration (s)
DEFAULT_DELAY = 60e-3    # Pulse delay (s)

# Sperical winder parameters
R0 = 1.8 # (m)

# CFL for stability of the numerical solver. To ensure waves dont travel faster than numerical solver can track it.
CFL = (C0 * DEFAULT_DT) / DEFAULT_DX

# Wind Advection
WIND_SPEED = 5.0        # m/s
WIND_DIR_DEG = 0.0      # 0 degrees = +Z (Vertical), 90 degrees = +X (Horizontal)