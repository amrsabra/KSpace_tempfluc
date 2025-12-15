import numpy as np
from numba import njit, prange
import constants


@njit(parallel=True, fastmath=True)
def accumulate_core(
    p_surf,
    u_surf,
    surface_normals,
    time_idx_floor,
    time_weight,
    geometric_weights,
    far_field_buffer,
    dx,
    c0,
    rho0,
    n_time_steps,
    time_step,
):
    """Core NTFF accumulation loop (Eq. 17).

    Notes
    -----
    * Only the outer loop (over directions) is parallelised.
      The inner loop (over surface points) is serial to avoid
      race conditions when multiple surface points contribute
      to the same (direction, time) bin in ``far_field_buffer``.
    * The time indexing follows the paper's definition of
      t0 = t + (r_hat · r') / c0, implemented via a precomputed
      integer floor offset (time_idx_floor) and linear
      interpolation weight (time_weight).
    """
    n_dirs, n_surface_points = geometric_weights.shape
    dS = dx * dx

    for i_dir in prange(n_dirs):
        # serial loop over surface points to avoid concurrent writes
        for i_pt in range(n_surface_points):
            # Surface normal at this point
            n_hat = surface_normals[i_pt]

            # Retarded time index (integer part)
            t_target = time_step - time_idx_floor[i_dir, i_pt]
            if t_target < 0 or t_target >= n_time_steps:
                continue

            # Pressure contribution: (n̂ · r̂) p / c0
            n_dot_r = geometric_weights[i_dir, i_pt]
            pressure_contrib = n_dot_r * p_surf[i_pt] / c0

            # Velocity contribution: ρ0 (n̂ · u)
            n_dot_u = (
                n_hat[0] * u_surf[i_pt, 0]
                + n_hat[1] * u_surf[i_pt, 1]
                + n_hat[2] * u_surf[i_pt, 2]
            )
            velocity_contrib = rho0 * n_dot_u

            integrand = pressure_contrib + velocity_contrib

            # Linear interpolation in time between neighbouring bins
            w = time_weight[i_dir, i_pt]
            far_field_buffer[i_dir, t_target] += integrand * dS * (1.0 - w)
            if t_target + 1 < n_time_steps:
                far_field_buffer[i_dir, t_target + 1] += integrand * dS * w


class NTFFTransform:
    """Near-to-far-field (NTFF) transform for the scattered field.

    This implements Eq. (17) of the paper for a cubic surface S
    surrounding the scattering volume, placed just inside the PML.
    """

    def __init__(self, grid_x, grid_y, grid_z, dx, c0, rho0, pml_depth=constants.PML_DEPTH):
        self.N = len(grid_x)
        self.dx = dx
        self.c0 = c0
        self.rho0 = rho0
        self.pml_depth = pml_depth

        # Index of the inner PML boundary where the NTFF surface is placed
        self.surface_idx = pml_depth
        self._create_surface_grid(grid_x, grid_y, grid_z)

    def _create_surface_grid(self, grid_x, grid_y, grid_z):
        idx = self.surface_idx  # one cell inside the PML
        N = self.N
        interior_slice = slice(idx, N - idx)

        x_int = grid_x[interior_slice]
        y_int = grid_y[interior_slice]
        z_int = grid_z[interior_slice]

        self.surface_points = []   # r'
        self.surface_normals = []  # n̂
        self.surface_indices = []  # (i, j, k) indices into 3D arrays

        # Face 1: x_min (left, normal = -x)
        x_face = grid_x[idx]
        for j, y in enumerate(y_int):
            for k, z in enumerate(z_int):
                self.surface_points.append([x_face, y, z])
                self.surface_normals.append([-1.0, 0.0, 0.0])
                self.surface_indices.append([idx, idx + j, idx + k])

        # Face 2: x_max (right, normal = +x)
        x_face = grid_x[N - idx - 1]
        for j, y in enumerate(y_int):
            for k, z in enumerate(z_int):
                self.surface_points.append([x_face, y, z])
                self.surface_normals.append([1.0, 0.0, 0.0])
                self.surface_indices.append([N - idx - 1, idx + j, idx + k])

        # Face 3: y_min (front, normal = -y)
        y_face = grid_y[idx]
        for i, x in enumerate(x_int):
            for k, z in enumerate(z_int):
                self.surface_points.append([x, y_face, z])
                self.surface_normals.append([0.0, -1.0, 0.0])
                self.surface_indices.append([idx + i, idx, idx + k])

        # Face 4: y_max (back, normal = +y)
        y_face = grid_y[N - idx - 1]
        for i, x in enumerate(x_int):
            for k, z in enumerate(z_int):
                self.surface_points.append([x, y_face, z])
                self.surface_normals.append([0.0, 1.0, 0.0])
                self.surface_indices.append([idx + i, N - idx - 1, idx + k])

        # Face 5: z_min (bottom, normal = -z)
        z_face = grid_z[idx]
        for i, x in enumerate(x_int):
            for j, y in enumerate(y_int):
                self.surface_points.append([x, y, z_face])
                self.surface_normals.append([0.0, 0.0, -1.0])
                self.surface_indices.append([idx + i, idx + j, idx])

        # Face 6: z_max (top, normal = +z)
        z_face = grid_z[N - idx - 1]
        for i, x in enumerate(x_int):
            for j, y in enumerate(y_int):
                self.surface_points.append([x, y, z_face])
                self.surface_normals.append([0.0, 0.0, 1.0])
                self.surface_indices.append([idx + i, idx + j, N - idx - 1])

        # Convert lists to arrays
        self.surface_points = np.array(self.surface_points, dtype=np.float64)
        self.surface_normals = np.array(self.surface_normals, dtype=np.float64)
        self.surface_indices = np.array(self.surface_indices, dtype=np.int64)

        # Optional stride (keep every 'stride'-th surface point)
        stride = 1
        self.surface_points = self.surface_points[::stride]
        self.surface_normals = self.surface_normals[::stride]
        self.surface_indices = self.surface_indices[::stride]

        self.n_surface_points = len(self.surface_points)
        print(f"NTFF: Created surface with {self.n_surface_points} points on 6 faces")

    def precompute_coefficients(self, far_field_directions, dt):
        """Precompute geometric and temporal weights for the NTFF.

        Parameters
        ----------
        far_field_directions : array_like, shape (n_dirs, 3)
            Unit vectors \hat{r} giving the far-field directions.
        dt : float
            Time step, used to convert retardation times into
            discrete indices and interpolation weights.
        """
        self.dt = dt
        self.directions = np.asarray(far_field_directions, dtype=np.float64)
        self.n_dirs = self.directions.shape[0]

        pts = self.surface_points  # (n_pts, 3)
        norms = self.surface_normals  # (n_pts, 3)

        # r_hat · r'  →  (n_dirs, 3) @ (3, n_pts) = (n_dirs, n_pts)
        self.retardation_times = (self.directions @ pts.T) / self.c0

        # n_hat · r_hat  →  (n_dirs, 3) @ (3, n_pts) = (n_dirs, n_pts)
        self.geometric_weights = self.directions @ norms.T

        # Convert retardation times to time indices and interpolation weights
        self.time_indices = self.retardation_times / dt
        self.time_idx_floor = np.floor(self.time_indices).astype(np.int64)
        self.time_weight = self.time_indices - self.time_idx_floor

        max_offset = np.max(np.abs(self.time_idx_floor))
        print(f"NTFF: Coefficients computed for {self.n_dirs} directions")
        print(f"      Max time offset: {max_offset} steps ({max_offset * dt * 1e6:.2f} μs)")

    def initialize_buffer(self, n_time_steps):
        """Allocate and zero the far-field accumulation buffer."""
        self.far_field_buffer = np.zeros((self.n_dirs, n_time_steps), dtype=np.float64)
        self.n_time_steps = n_time_steps
        print(f"NTFF: Buffer initialized ({self.n_dirs} × {n_time_steps})")

    def accumulate(self, p_s, ux_s, uy_s, uz_s, time_step):
        """Accumulate contributions from the current near-field state.

        Parameters
        ----------
        p_s : ndarray, shape (N, N, N)
            Total scattered pressure on the grid.
        ux_s, uy_s, uz_s : ndarray, shape (N, N, N)
            Scattered velocity components on the grid.
        time_step : int
            Current time-step index.
        """
        idx_i = self.surface_indices[:, 0]
        idx_j = self.surface_indices[:, 1]
        idx_k = self.surface_indices[:, 2]

        # Gather pressure at surface points
        p_surf = p_s[idx_i, idx_j, idx_k]

        # Gather velocity at surface points into a (n_pts, 3) array
        u_surf = np.empty((self.n_surface_points, 3), dtype=p_s.dtype)
        u_surf[:, 0] = ux_s[idx_i, idx_j, idx_k]
        u_surf[:, 1] = uy_s[idx_i, idx_j, idx_k]
        u_surf[:, 2] = uz_s[idx_i, idx_j, idx_k]

        accumulate_core(
            p_surf,
            u_surf,
            self.surface_normals,
            self.time_idx_floor,
            self.time_weight,
            self.geometric_weights,
            self.far_field_buffer,
            self.dx,
            self.c0,
            self.rho0,
            self.n_time_steps,
            time_step,
        )

    def compute_far_field(self):
        """Apply the time derivative to the accumulated buffer (Eq. 17).

        Returns
        -------
        p_ff : ndarray, shape (n_dirs, n_time_steps)
            Far-field pressure as a function of angle and time.
        """
        p_ff = np.zeros_like(self.far_field_buffer)

        # Central differences for interior time indices
        for i in range(1, self.n_time_steps - 1):
            p_ff[:, i] = (
                self.far_field_buffer[:, i + 1] - self.far_field_buffer[:, i - 1]
            ) / (2.0 * self.dt)

        # One-sided differences at boundaries
        p_ff[:, 0] = (
            self.far_field_buffer[:, 1] - self.far_field_buffer[:, 0]
        ) / self.dt
        p_ff[:, -1] = (
            self.far_field_buffer[:, -1] - self.far_field_buffer[:, -2]
        ) / self.dt

        return p_ff
