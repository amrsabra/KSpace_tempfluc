# debug_simulation.py
import numpy as np
import constants
from simulator import KSpaceAcousticScattering

def run_debug():
    print("--- STARTING DEBUG SIMULATION ---")
    N = constants.DEFAULT_N
    dx = constants.DEFAULT_DX
    dt = constants.DEFAULT_DT

    sim = KSpaceAcousticScattering(N=N, dx=dx, dt=dt)
    
    print("1. Creating Atmosphere...")
    T, window, V_scat = sim.create_bragg_atmosphere(
        fm=constants.DEFAULT_FM, 
        DT=1.0, 
        r0=constants.R0
    )
    
    # Check T statistics
    print(f"   T stats: min={np.min(T):.4f}, max={np.max(T):.4f}, mean={np.mean(T):.4f}")
    
    # Check if T touches the PML (Stability Risk)
    pml_idx = constants.PML_DEPTH
    T_in_pml = T[:pml_idx, :, :]
    if np.any(np.abs(T_in_pml - constants.T0) > 1e-5):
        print("   WARNING: Temperature fluctuations detected INSIDE the PML region!")
        print("            This causes undefined behavior in Split-Field PML.")
    else:
        print("   OK: PML region is homogeneous (T=T0).")

    print("2. Starting Time Loop...")
    
    # Manually run the loop from simulator.py to inspect variables
    sim.ntff.initialize_buffer(100)
    sim.pml.set_dt(dt)
    
    # Init fields
    px_s = np.zeros((N, N, N))
    py_s = np.zeros((N, N, N))
    pz_s = np.zeros((N, N, N))
    ux_s = np.zeros((N, N, N))
    uy_s = np.zeros((N, N, N))
    uz_s = np.zeros((N, N, N))
    
    ux_s_prev = np.copy(ux_s)
    uy_s_prev = np.copy(uy_s)
    uz_s_prev = np.copy(uz_s)
    px_s_prev = np.copy(px_s)
    py_s_prev = np.copy(py_s)
    pz_s_prev = np.copy(pz_s)
    
    rho = constants.RHO0 * constants.T0 / T
    _, u_i_z_prev = sim.incident.plane_wave(-dt)

    for step in range(100):
        t = step * dt
        
        # 1. Incident Wave
        p_i, u_i_z = sim.incident.plane_wave(t)
        duiz_dt = (u_i_z - u_i_z_prev) / dt
        u_i_z_prev = u_i_z
        
        # 2. Source Term
        source_term = (constants.RHO0 - rho) * duiz_dt
        source_max = np.max(np.abs(source_term))
        
        # 3. Calculate Gradients
        p_s = px_s + py_s + pz_s
        dpdx, dpdy, dpdz = sim.kspace_ops.derivatives_xyz(p_s)
        
        # 4. RHS Calculation (The Stability Fix Logic)
        rho_const_inv = 1.0 / constants.RHO0
        rho_inv = 1.0 / rho
        corr = rho_const_inv - rho_inv
        
        rhs_ux = -dpdx * rho_const_inv + corr * dpdx
        rhs_uy = -dpdy * rho_const_inv + corr * dpdy
        rhs_uz = -dpdz * rho_const_inv + corr * dpdz + (source_term / rho)
        
        # 5. Update Velocity
        ux_s = sim.pml.update_velocity_component(ux_s_prev, rhs_ux, dt, 'x')
        uy_s = sim.pml.update_velocity_component(uy_s_prev, rhs_uy, dt, 'y')
        uz_s = sim.pml.update_velocity_component(uz_s_prev, rhs_uz, dt, 'z')
        ux_s_prev, uy_s_prev, uz_s_prev = ux_s, uy_s, uz_s
        
        # 6. Update Pressure
        duxdx = sim.kspace_ops.derivative(ux_s, 'x')
        duydy = sim.kspace_ops.derivative(uy_s, 'y')
        duzdz = sim.kspace_ops.derivative(uz_s, 'z')
        
        rhs_px = -constants.RHO0 * constants.C0**2 * duxdx
        rhs_py = -constants.RHO0 * constants.C0**2 * duydy
        rhs_pz = -constants.RHO0 * constants.C0**2 * duzdz
        
        px_s = sim.pml.update_pressure_component(px_s_prev, rhs_px, dt, 'x')
        py_s = sim.pml.update_pressure_component(py_s_prev, rhs_py, dt, 'y')
        pz_s = sim.pml.update_pressure_component(pz_s_prev, rhs_pz, dt, 'z')
        px_s_prev, py_s_prev, pz_s_prev = px_s, py_s, pz_s
        
        # MONITORING
        max_p = np.max(np.abs(p_s))
        max_u = np.max(np.abs(ux_s))
        
        print(f"Step {step}: Max P={max_p:.2e}, Max U={max_u:.2e}, Source={source_max:.2e}")
        
        if max_p > 1e10 or np.isnan(max_p):
            print("!!! EXPLOSION DETECTED !!!")
            break

if __name__ == "__main__":
    run_debug()