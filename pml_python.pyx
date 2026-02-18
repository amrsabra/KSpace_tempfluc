import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)  # Deactivates bounds checking for speed
@cython.wraparound(False)   # Deactivates negative indexing
@cython.nonecheck(False)
@cython.cdivision(True)     # Enables faster C-style division

def update_pml_c(double[:, :, :] arr, 
                    double[:, :, :] rhs, 
                    double[:] exp_half, 
                    double[:] exp_neg_half, 
                    double dt, 
                    str direction):
    """
    Compiled C-kernel for Equation 13.
    Updates the 3D grid in-place using 1D absorption profiles.
    """
    cdef int Ni = arr.shape[0]
    cdef int Nj = arr.shape[1]
    cdef int Nk = arr.shape[2]
    cdef int i, j, k
    
    # Logic: u_new = (u_prev * exp_neg_half + dt * rhs) / exp_half
    if direction == 'x':
        for i in range(Ni):
            for j in range(Nj):
                for k in range(Nk):
                    arr[i, j, k] = (arr[i, j, k] * exp_neg_half[i] + dt * rhs[i, j, k]) / exp_half[i]
    
    elif direction == 'y':
        for i in range(Ni):
            for j in range(Nj):
                for k in range(Nk):
                    arr[i, j, k] = (arr[i, j, k] * exp_neg_half[j] + dt * rhs[i, j, k]) / exp_half[j]
                    
    elif direction == 'z':
        for i in range(Ni):
            for j in range(Nj):
                for k in range(Nk):
                    arr[i, j, k] = (arr[i, j, k] * exp_neg_half[k] + dt * rhs[i, j, k]) / exp_half[k]