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
    time_step
):
    n_dirs, n_surface_points = geometric_weights.shape
    dS = dx * dx

    for i_dir in prange(n_dirs):
        for i_pt in prange(n_surface_points):
            # normal at this surface point
            n_hat = surface_normals[i_pt]

            # retarded time index
            t_target = time_step - time_idx_floor[i_dir, i_pt]
            if t_target < 0 or t_target >= n_time_steps:
                continue

            # pressure part
            n_dot_r = geometric_weights[i_dir, i_pt]
            pressure_contrib = n_dot_r * p_surf[i_pt] / c0

            # velocity part: n·u
            n_dot_u = (
                n_hat[0] * u_surf[i_pt, 0]
                + n_hat[1] * u_surf[i_pt, 1]
                + n_hat[2] * u_surf[i_pt, 2]
            )
            velocity_contrib = rho0 * n_dot_u

            integrand = pressure_contrib + velocity_contrib

            # linear interpolation in time
            w = time_weight[i_dir, i_pt]
            far_field_buffer[i_dir, t_target] += integrand * dS * (1.0 - w)
            if t_target + 1 < n_time_steps:
                far_field_buffer[i_dir, t_target + 1] += integrand * dS * w


class NTFFTransform:
    
    def __init__(self, grid_x, grid_y, grid_z, dx, c0, rho0, pml_depth=constants.PML_DEPTH):
        self.N = len(grid_x)
        self.dx = dx
        self.c0 = c0
        self.rho0 = rho0
        self.pml_depth = pml_depth
        
        self.surface_idx = pml_depth
        self._create_surface_grid(grid_x, grid_y, grid_z)
        
    def _create_surface_grid(self, grid_x, grid_y, grid_z):
        idx = self.surface_idx # PML depth so we can stay outside 
        N = self.N
        interior_slice = slice(idx, N - idx)
        
        x_int = grid_x[interior_slice] # coordinates of interior region 
        y_int = grid_y[interior_slice]
        z_int = grid_z[interior_slice]
        
        self.surface_points = [] # rprime
        self.surface_normals = [] # ncap
        self.surface_indices = [] # (i,j,k) so we can read p and u from 3D arrays
        
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
        
        self.surface_points = np.array(self.surface_points) # convert list to an array
        self.surface_normals = np.array(self.surface_normals)
        self.surface_indices = np.array(self.surface_indices, dtype=int)

        stride = 1 # its like "every how much surface points do we keep measurements"
        self.surface_points = self.surface_points[::stride]
        self.surface_normals = self.surface_normals[::stride]
        self.surface_indices = self.surface_indices[::stride]

        self.n_surface_points = len(self.surface_points)
        
        print(f"NTFF: Created surface with {self.n_surface_points} points on 6 faces")
        
    def precompute_coefficients(self, far_field_directions, dt): #vectorised
        self.dt = dt
        self.directions = np.asarray(far_field_directions, dtype=np.float64)
        self.n_dirs = self.directions.shape[0]

        # (n_pts, 3)
        pts = self.surface_points.astype(np.float64)
        norms = self.surface_normals.astype(np.float64)

        # r_hat · r'  →  (n_dirs, 3) @ (3, n_pts) = (n_dirs, n_pts)
        self.retardation_times = (self.directions @ pts.T) / self.c0

        # n_hat · r_hat  →  (n_dirs, 3) @ (3, n_pts) = (n_dirs, n_pts)
        # same shapes, just different physical meaning
        self.geometric_weights = self.directions @ norms.T

        # convert to time indices
        self.time_indices = self.retardation_times / dt
        self.time_idx_floor = np.floor(self.time_indices).astype(np.int64)
        self.time_weight = self.time_indices - self.time_idx_floor

        max_offset = np.max(np.abs(self.time_idx_floor))
        print(f"NTFF: Coefficients computed for {self.n_dirs} directions")
        print(f"      Max time offset: {max_offset} steps ({max_offset*dt*1e6:.2f} μs)")

        
    def initialize_buffer(self, n_time_steps):
        self.far_field_buffer = np.zeros((self.n_dirs, n_time_steps))
        self.n_time_steps = n_time_steps
        print(f"NTFF: Buffer initialized ({self.n_dirs} × {n_time_steps})")
        
    def accumulate(self, p_s, ux_s, uy_s, uz_s, time_step):
        # Vectorised gather from the 3D arrays at all surface points

        idx_i = self.surface_indices[:, 0]
        idx_j = self.surface_indices[:, 1]
        idx_k = self.surface_indices[:, 2]

        p_surf = p_s[idx_i, idx_j, idx_k]

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
        # Apply time derivative EQN 17
        p_ff = np.zeros_like(self.far_field_buffer)
        
        # Central difference
        for i in range(1, self.n_time_steps - 1):
            p_ff[:, i] = (self.far_field_buffer[:, i+1] - 
                         self.far_field_buffer[:, i-1]) / (2 * self.dt)
        
        # Forward/backward difference at boundaries
        p_ff[:, 0] = (self.far_field_buffer[:, 1] - self.far_field_buffer[:, 0]) / self.dt
        p_ff[:, -1] = (self.far_field_buffer[:, -1] - self.far_field_buffer[:, -2]) / self.dt
        
        return p_ff