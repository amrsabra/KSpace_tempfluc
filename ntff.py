import numpy as np
from numba import njit, prange
import constants

# Numba library compiled to machine code for speed and runs across multiple CPU cores in parallel.
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
    Meaning, it calculates the geometry (dot products, distances, and retardation times) during the sim at every time step.
    Replaces the massive lookup table with compute ops.
    look up table pre-calculate distance and angle between every point, but takes too much RAM.
    """
    n_dirs = directions.shape[0] # we find number of directions from the directions arrays.
    n_surface_points = surface_points.shape[0] # we find number of surface points from the arrays.
    dS = dx * dx # dr′ or dS
    
    # prange form Numba, allows us to execute a loop in parallel.
    for i_dir in prange(n_dirs):
        
        # Pre-load direction vector for this thread
        rx = directions[i_dir, 0]
        ry = directions[i_dir, 1]
        rz = directions[i_dir, 2]
        
        for i_pt in range(n_surface_points):# i_pt is index of surface point 
            # 1. Geometry Calculation (Replaces Lookup Table)
            # Surface Point
            px = surface_points[i_pt, 0] # i_pt is the row index, representing one of the 360 arrays to put data in.
            py = surface_points[i_pt, 1] # 0, 1, 2 show the component of the unit vector r′ (x,y,z)
            pz = surface_points[i_pt, 2]
            
            # Surface Normal
            nx = surface_normals[i_pt, 0]
            ny = surface_normals[i_pt, 1]
            nz = surface_normals[i_pt, 2]
            
            # Dot Products
            # r_hat (r) . r' (p)(Surface point), coordinates of specific point.
            # r_dot_p shows relative distance of surface point to the center of grid
            r_dot_p = rx*px + ry*py + rz*pz
            
            # n_hat . r_hat (Geometric weight)
            # n_hat is the unit normal vector pointing straight out of surfact point
            n_dot_r = nx*rx + ny*ry + nz*rz
            
            # 2. Time Indexing
            # retardation (time) = (r_hat . r') / c0
            # t_ret = t - (r_hat . r')/c0
            # We map this to indices.
            retardation_val = r_dot_p / c0
            retardation_idx = retardation_val / dt # count of timesteps
            
            # Floor and Weight
            idx_floor = int(np.floor(retardation_idx)) # to make retardation_idx an exact int.
            w = retardation_idx - idx_floor # remaining that is discarded to get int.
            
            # Target Time Bin
            # aligning the sim time (t) with ff recording time (t')
            t_target = time_step - idx_floor
            
            # Boundary Check (just a safety net to ensure sound waves reach before recording scattering)
            if t_target < 0 or t_target >= n_time_steps - 1:
                continue

            # 3. Physics Accumulation
            # Pressure term: (n . r) * p / c0
            p_val = p_surf[i_pt]
            term_p = n_dot_r * p_val / c0
            
            # Velocity term: rho0 * (n . u)
            u_dot_n = ux_surf[i_pt]*nx + uy_surf[i_pt]*ny + uz_surf[i_pt]*nz
            term_u = rho0 * u_dot_n
            
            integrand = (term_p + term_u) * dS # EQN 17

            # Linear Interpolation update
            '''
            "travel time" from a point to the far-field box rarely lands perfectly on a whole time step.
            eg: a delay of 5.4 steps, so integrand is split into two, 1 - w to get majority of signal,
            and t_target + 1 to get remainder of signal using w
            '''
            far_field_buffer[i_dir, t_target] += integrand * (1.0 - w) # i_dir is the direction
            far_field_buffer[i_dir, t_target + 1] += integrand * w # t_target is time slot of the recording for gathering  "data"


class NTFFTransform:
    def __init__(self, grid_x, grid_y, grid_z, dx, c0, rho0, pml_depth=constants.PML_DEPTH):
        self.dx = dx
        self.c0 = c0
        self.rho0 = rho0
        self.pml_depth = pml_depth
        
        # Setup Surface Geometry
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
        
        # Helper to add face one at a time.
        def add_face(i_range, j_range, k_range, norm):
            # Create meshgrid for current face
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

        # Merge to master array
        self.surface_indices = np.vstack(indices).astype(np.int64)
        self.surface_points = np.vstack(points).astype(np.float64)
        self.surface_normals = np.vstack(normals).astype(np.float64)
        
        self.n_surface_points = len(self.surface_points)
        print(f"NTFF: Created surface with {self.n_surface_points} points on 6 faces")

    def precompute_coefficients(self, far_field_directions, dt):

        # Store simulation parameters. 
        self.dt = dt
        self.directions = np.asarray(far_field_directions, dtype=np.float64)

    def initialize_buffer(self, n_time_steps):
        # Allocate and zero the far-field accumulation buffer.
        if self.directions is None:
            raise ValueError("NTFF directions not set. Call precompute_coefficients first.")
            
        self.n_time_steps = n_time_steps
        self.far_field_buffer = np.zeros((len(self.directions), n_time_steps), dtype=np.float64)
        print(f"NTFF: Buffer initialized ({len(self.directions)} x {n_time_steps})")

    def accumulate(self, p_s, ux_s, uy_s, uz_s, time_step):
        # Accumulate contributions from the current near-field state.
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
        # Apply the time derivative to the accumulated buffer (Eq. 17).
        p_ff = np.zeros_like(self.far_field_buffer)

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