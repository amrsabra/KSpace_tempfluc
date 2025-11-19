import numpy as np
import matplotlib.pyplot as plt
from simulator import KSpaceAcousticScattering
import constants
from constants import R0, DEFAULT_N, DEFAULT_DX, DEFAULT_DT, PML_DEPTH, DEFAULT_FM

def generate_figure_3():
    print("\n=== Generating Figure 3 ===")
    
    sim = KSpaceAcousticScattering(N=DEFAULT_N, dx=DEFAULT_DX, dt=DEFAULT_DT)
    
    r0_values = [0.15, 0.3, 0.6, 1.2, 2.4]
    fm = DEFAULT_FM
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    results = []
    
    for idx, r0 in enumerate(r0_values):
        print(f"\nSimulating r0 = {r0} m")
        
        T, window ,V_scat = sim.create_bragg_atmosphere(fm, DT=1.0, r0=r0) #DT here is temp fluctuation amplitude.
        far_field, angles = sim.simulate_scattering(T, fm=fm)
        H_dB = sim.calculate_scattering_cross_section(far_field)
        
        angles_rad = np.deg2rad(angles)
        axes[idx].plot(angles_rad, H_dB - np.max(H_dB), 'b-', linewidth=2)
        axes[idx].set_theta_zero_location('N')
        axes[idx].set_theta_direction(-1)
        axes[idx].set_ylim(-60, 0)
        axes[idx].set_title(f'r₀ = {r0} m', fontsize=12, pad=20)
        axes[idx].grid(True)
        
        results.append({'r0': r0, 'V_scat': V_scat, 'H_dB': H_dB})
    
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig('figure3_bragg_scattering.png', dpi=150, bbox_inches='tight')
    print("\nSaved: figure3_bragg_scattering.png")
    
    return results

def generate_figure_6():
    print("\n=== Generating Figure 6 ===")
    
    sim = KSpaceAcousticScattering(N=DEFAULT_N, dx=DEFAULT_DX, dt=DEFAULT_DT)
    CT2 = 1.5e-7 * constants.T0**2
    T, window, V_scat = sim.create_kolmogorov_atmosphere(CT2, r0=R0, seed=42)
    
    mid_y = sim.N // 2
    T_slice = T[:, mid_y, :] - constants.T0
    
    print(f"Temperature statistics:")
    print(f"  Std Dev: {np.std(T_slice):.4f} K")
    print(f"  Range: [{np.min(T_slice):.4f}, {np.max(T_slice):.4f}] K")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    extent = [-sim.domain_size/2, sim.domain_size/2, -sim.domain_size/2, sim.domain_size/2]
    vmax = max(abs(np.min(T_slice)), abs(np.max(T_slice)), 0.5)
    
    im = ax.imshow(T_slice.T, extent=extent, origin='lower', cmap='RdBu_r', 
                   vmin=-vmax, vmax=vmax, aspect='equal', interpolation='bilinear')
    
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('z (m)', fontsize=12)
    ax.set_title('Temperature Fluctuation Field (Kolmogorov Spectrum)', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Temperature - T₀ (K)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('figure6_temperature_field.png', dpi=150, bbox_inches='tight')
    print("\nSaved: figure6_temperature_field.png")

def generate_figure_7():
    print("\n=== Generating Figure 7 ===")
    
    sim = KSpaceAcousticScattering(N=DEFAULT_N, dx=DEFAULT_DX, dt=DEFAULT_DT)
    
    CT2 = 1.5e-7 * constants.T0**2
    r0 = R0
    fm = DEFAULT_FM
    n_realizations = PML_DEPTH
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    all_H_dB = []
    
    for i in range(n_realizations):
        print(f"\nRealization {i+1}/{n_realizations}")
        
        T, V_scat = sim.create_kolmogorov_atmosphere(CT2, r0, seed=i*42)
        far_field, angles = sim.simulate_scattering(T, fm=fm)
        H_dB = sim.calculate_scattering_cross_section(far_field)
        
        all_H_dB.append(H_dB)
        angles_rad = np.deg2rad(angles)
        ax.plot(angles_rad, H_dB - np.max(H_dB), 'gray', alpha=0.3, linewidth=1)
    
    H_dB_avg = np.mean(all_H_dB, axis=0)
    ax.plot(angles_rad, H_dB_avg - np.max(H_dB_avg), 'k-', linewidth=3, label='Average')
    
    # Analytical model EQN 23
    k = 2 * np.pi * fm / constants.C0
    theta_rad = np.deg2rad(angles)
    theta_safe = np.copy(theta_rad)
    theta_safe[np.abs(theta_safe) < 0.1] = 0.1
    
    sigma = (0.0041 * (CT2 / constants.T0**2) * k**(1/3) * 
             np.cos(theta_safe)**2 / (np.sin(np.abs(theta_safe)/2))**(11/3))
    H_analytical = (4 * np.pi)**2 * V_scat * sigma
    H_analytical_dB = 10 * np.log10(H_analytical / np.max(H_analytical) + 1e-12)
    
    ax.plot(angles_rad, H_analytical_dB, 'r--', linewidth=2, label='Analytical')
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(-60, 0)
    ax.set_title('Kolmogorov Atmosphere Scattering', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('figure7_kolmogorov_scattering.png', dpi=150, bbox_inches='tight')
    print("\nSaved: figure7_kolmogorov_scattering.png")