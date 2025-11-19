import numpy as np
from constants import PML_DEPTH

class NTFFTransform:
    
    def __init__(self, grid_x, grid_y, grid_z, dx, c0, rho0, pml_depth=PML_DEPTH):
        self.N = len(grid_x)
        self.dx = dx
        self.c0 = c0
        self.rho0 = rho0
        self.pml_depth = pml_depth
        
        self.surface_idx = pml_depth
        self._create_surface_grid(grid_x, grid_y, grid_z)
        
    def _create_surface_grid(self, grid_x, grid_y, grid_z):
        idx = self.surface_idx
        N = self.N
        interior_slice = slice(idx, N - idx)
        
        x_int = grid_x[interior_slice]
        y_int = grid_y[interior_slice]
        z_int = grid_z[interior_slice]
        
        self.surface_points = []
        self.surface_normals = []
        self.surface_indices = []
        
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
        
        self.surface_points = np.array(self.surface_points)
        self.surface_normals = np.array(self.surface_normals)
        self.surface_indices = np.array(self.surface_indices, dtype=int)

        # Thin the NTFF surface to reduce cost
        stride = 4  # try 4 first, you can change to 2 or 3 if needed
        self.surface_points = self.surface_points[::stride]
        self.surface_normals = self.surface_normals[::stride]
        self.surface_indices = self.surface_indices[::stride]

        self.n_surface_points = len(self.surface_points)
        
        print(f"NTFF: Created surface with {self.n_surface_points} points on 6 faces")
        
    def precompute_coefficients(self, far_field_directions, dt):
        n_dirs = len(far_field_directions)
        self.dt = dt
        self.directions = far_field_directions
        self.n_dirs = n_dirs
        
        self.retardation_times = np.zeros((n_dirs, self.n_surface_points))
        self.geometric_weights = np.zeros((n_dirs, self.n_surface_points))
        
        for i_dir, r_hat in enumerate(far_field_directions):
            for i_pt in range(self.n_surface_points):
                r_prime = self.surface_points[i_pt]
                n_hat = self.surface_normals[i_pt]
                
                # EQN 17
                self.retardation_times[i_dir, i_pt] = np.dot(r_hat, r_prime) / self.c0
                
                # EQN 17
                self.geometric_weights[i_dir, i_pt] = np.dot(n_hat, r_hat)
        
        self.time_indices = self.retardation_times / dt
        self.time_idx_floor = np.floor(self.time_indices).astype(int)
        self.time_weight = self.time_indices - self.time_idx_floor
        
        max_offset = np.max(np.abs(self.time_idx_floor))
        print(f"NTFF: Coefficients computed for {n_dirs} directions")
        print(f"      Max time offset: {max_offset} steps ({max_offset*dt*1e6:.2f} μs)")
        
    def initialize_buffer(self, n_time_steps):
        self.far_field_buffer = np.zeros((self.n_dirs, n_time_steps))
        self.n_time_steps = n_time_steps
        print(f"NTFF: Buffer initialized ({self.n_dirs} × {n_time_steps})")
        
    def accumulate(self, p_s, ux_s, uy_s, uz_s, time_step):
        # Extract surface data
        p_surf = np.zeros(self.n_surface_points)
        u_surf = np.zeros((self.n_surface_points, 3))
        
        for i_pt in range(self.n_surface_points):
            i, j, k = self.surface_indices[i_pt]
            p_surf[i_pt] = p_s[i, j, k]
            u_surf[i_pt, 0] = ux_s[i, j, k]
            u_surf[i_pt, 1] = uy_s[i, j, k]
            u_surf[i_pt, 2] = uz_s[i, j, k]
        
        # Accumulate for each direction
        for i_dir in range(self.n_dirs):
            for i_pt in range(self.n_surface_points):
                n_hat = self.surface_normals[i_pt]
                
                t_target = time_step - self.time_idx_floor[i_dir, i_pt]
                
                if t_target < 0 or t_target >= self.n_time_steps:
                    continue
                
                # EQN 17
                n_dot_r = self.geometric_weights[i_dir, i_pt]
                pressure_contrib = n_dot_r * p_surf[i_pt] / self.c0
                
                n_dot_u = np.dot(n_hat, u_surf[i_pt])
                velocity_contrib = self.rho0 * n_dot_u
                
                integrand = pressure_contrib + velocity_contrib
                dS = self.dx * self.dx
                
                # Linear interpolation in time
                weight = self.time_weight[i_dir, i_pt]
                self.far_field_buffer[i_dir, t_target] += integrand * dS * (1.0 - weight)
                
                if t_target + 1 < self.n_time_steps:
                    self.far_field_buffer[i_dir, t_target + 1] += integrand * dS * weight
        
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