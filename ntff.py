import numpy as np
from numba import njit, prange
import constants

@njit(parallel=True, fastmath=True)
def accumulate_core_on_the_fly(
    p_surf,
    ux_surf, uy_surf, uz_surf,
    surface_points,
    surface_normals,
    directions,
    far_field_buffer,
    dx,
    c0,
    rho0,
    n_time_steps,
    time_step,
    dt
):
    """
    Computes NTFF contributions on-the-fly to save RAM.
    Replaces the massive lookup table with compute ops.
    """
    n_dirs = directions.shape[0]
    n_surface_points = surface_points.shape[0]
    dS = dx * dx
    
    # Iterate over directions in parallel (Thread-Safe writing to far_field_buffer rows)
    for i_dir in prange(n_dirs):
        
        # Pre-load direction vector for this thread
        rx = directions[i_dir, 0]
        ry = directions[i_dir, 1]
        rz = directions[i_dir, 2]
        
        for i_pt in range(n_surface_points):
            # 1. Geometry Calculation (Replaces Lookup Table)
            # Surface Point
            px = surface_points[i_pt, 0]
            py = surface_points[i_pt, 1]
            pz = surface_points[i_pt, 2]
            
            # Surface Normal
            nx = surface_normals[i_pt, 0]
            ny = surface_normals[i_pt, 1]
            nz = surface_normals[i_pt, 2]
            
            # Dot Products
            # r_hat . r' (Retardation distance)
            r_dot_p = rx*px + ry*py + rz*pz
            
            # n_hat . r_hat (Geometric weight)
            # Note: For far-field Green's function, we need n_hat . r_hat term
            n_dot_r = nx*rx + ny*ry + nz*rz
            
            # 2. Time Indexing
            # retardation = (r_hat . r') / c0
            # t_ret = t - (r_hat . r')/c0
            # We map this to indices.
            retardation_val = r_dot_p / c0
            retardation_idx = retardation_val / dt
            
            # Floor and Weight
            idx_floor = int(np.floor(retardation_idx))
            w = retardation_idx - idx_floor
            
            # Target Time Bin
            # The paper defines t' = t + (r_hat . r')/c0
            # In discrete steps: k' = k + idx_floor
            # However, usually we populate buffer at [k - idx] or similar depending on sign convention.
            # Based on previous code: t_target = time_step - time_idx_floor
            t_target = time_step - idx_floor
            
            # Boundary Check
            if t_target < 0 or t_target >= n_time_steps - 1:
                continue

            # 3. Physics Accumulation
            # Pressure term: (n . r) * p / c0
            p_val = p_surf[i_pt]
            term_p = n_dot_r * p_val / c0
            
            # Velocity term: rho0 * (n . u)
            u_dot_n = ux_surf[i_pt]*nx + uy_surf[i_pt]*ny + uz_surf[i_pt]*nz
            term_u = rho0 * u_dot_n
            
            integrand = (term_p + term_u) * dS

            # Linear Interpolation update
            # far_field_buffer[i_dir, t] += integrand * (1-w)
            # far_field_buffer[i_dir, t+1] += integrand * w
            far_field_buffer[i_dir, t_target] += integrand * (1.0 - w)
            far_field_buffer[i_dir, t_target + 1] += integrand * w


class NTFFTransform:
    """Near-to-far-field (NTFF) transform optimized for memory efficiency.
    Calculates geometric coefficients on-the-fly rather than storing 
    massive lookup tables.
    """

    def __init__(self, grid_x, grid_y, grid_z, dx, c0, rho0, pml_depth=constants.PML_DEPTH):
        self.dx = dx
        self.c0 = c0
        self.rho0 = rho0
        self.pml_depth = pml_depth
        
        # Setup Surface Geometry
        # We only store the points and normals (~10MB total for N=256)
        # instead of the ~4GB required for the full coefficient table.
        self.surface_idx = pml_depth
        self._create_surface_grid(grid_x, grid_y, grid_z)
        
        self.far_field_buffer = None
        self.directions = None
        self.dt = None

    def _create_surface_grid(self, grid_x, grid_y, grid_z):
        idx = self.surface_idx
        Nx, Ny, Nz = len(grid_x), len(grid_y), len(grid_z)
        
        # Lists to gather data
        indices = []
        points = []
        normals = []
        
        # Helper to add face
        def add_face(i_range, j_range, k_range, norm):
            # Create meshgrid for this face
            I, J, K = np.meshgrid(i_range, j_range, k_range, indexing='ij')
            I, J, K = I.flatten(), J.flatten(), K.flatten()
            
            # Store Indices
            # Note: We stack them to be (n_pts, 3)
            face_indices = np.stack((I, J, K), axis=1)
            indices.append(face_indices)
            
            # Store Points
            face_pts = np.stack((grid_x[I], grid_y[J], grid_z[K]), axis=1)
            points.append(face_pts)
            
            # Store Normals
            n_pts = len(I)
            face_norms = np.tile(norm, (n_pts, 1))
            normals.append(face_norms)

        # Interior ranges
        inner_x = np.arange(idx, Nx - idx)
        inner_y = np.arange(idx, Ny - idx)
        inner_z = np.arange(idx, Nz - idx)
        
        # Face 1: x_min (-x)
        add_face([idx], inner_y, inner_z, [-1.0, 0.0, 0.0])
        # Face 2: x_max (+x)
        add_face([Nx-idx-1], inner_y, inner_z, [1.0, 0.0, 0.0])
        
        # Face 3: y_min (-y)
        add_face(inner_x, [idx], inner_z, [0.0, -1.0, 0.0])
        # Face 4: y_max (+y)
        add_face(inner_x, [Ny-idx-1], inner_z, [0.0, 1.0, 0.0])
        
        # Face 5: z_min (-z)
        add_face(inner_x, inner_y, [idx], [0.0, 0.0, -1.0])
        # Face 6: z_max (+z)
        add_face(inner_x, inner_y, [Nz-idx-1], [0.0, 0.0, 1.0])

        # Concatenate and convert to arrays
        self.surface_indices = np.vstack(indices).astype(np.int64)
        self.surface_points = np.vstack(points).astype(np.float64)
        self.surface_normals = np.vstack(normals).astype(np.float64)
        
        self.n_surface_points = len(self.surface_points)
        print(f"NTFF: Created surface with {self.n_surface_points} points on 6 faces")

    def precompute_coefficients(self, far_field_directions, dt):
        """
        Store simulation parameters. 
        Note: Unlike the previous version, this DOES NOT precompute the large matrices.
        """
        self.dt = dt
        self.directions = np.asarray(far_field_directions, dtype=np.float64)
        print(f"NTFF: Configured for {len(self.directions)} directions (On-the-fly mode).")

    def initialize_buffer(self, n_time_steps):
        """Allocate and zero the far-field accumulation buffer."""
        if self.directions is None:
            raise ValueError("NTFF directions not set. Call precompute_coefficients first.")
            
        self.n_time_steps = n_time_steps
        self.far_field_buffer = np.zeros((len(self.directions), n_time_steps), dtype=np.float64)
        print(f"NTFF: Buffer initialized ({len(self.directions)} × {n_time_steps})")

    def accumulate(self, p_s, ux_s, uy_s, uz_s, time_step):
        """Accumulate contributions from the current near-field state."""
        # Gather surface data using advanced indexing
        # This extracts the values at the surface indices into 1D arrays
        idx = self.surface_indices
        p_surf = p_s[idx[:,0], idx[:,1], idx[:,2]]
        ux_surf = ux_s[idx[:,0], idx[:,1], idx[:,2]]
        uy_surf = uy_s[idx[:,0], idx[:,1], idx[:,2]]
        uz_surf = uz_s[idx[:,0], idx[:,1], idx[:,2]]

        # Call the JIT-compiled kernel
        accumulate_core_on_the_fly(
            p_surf,
            ux_surf, uy_surf, uz_surf,
            self.surface_points,
            self.surface_normals,
            self.directions,
            self.far_field_buffer,
            self.dx,
            self.c0,
            self.rho0,
            self.n_time_steps,
            time_step,
            self.dt
        )

    def compute_far_field(self):
        """Apply the time derivative to the accumulated buffer (Eq. 17)."""
        p_ff = np.zeros_like(self.far_field_buffer)

        # Central differences for interior time indices
        p_ff[:, 1:-1] = (
            self.far_field_buffer[:, 2:] - self.far_field_buffer[:, :-2]
        ) / (2.0 * self.dt)

        # One-sided differences at boundaries
        p_ff[:, 0] = (
            self.far_field_buffer[:, 1] - self.far_field_buffer[:, 0]
        ) / self.dt
        p_ff[:, -1] = (
            self.far_field_buffer[:, -1] - self.far_field_buffer[:, -2]
        ) / self.dt

        return p_ff