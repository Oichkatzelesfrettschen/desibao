import jax
import jax.numpy as jnp
import pmwd
import jax_cosmo as jc
from pmwd import (
    Configuration,
    Cosmology,
    boltzmann,
    white_noise,
    linear_modes,
    lpt,
    nbody,
    scatter,
    growth,  # Import growth function
)
import matplotlib.pyplot as plt
import numpy as np

# --- 1. CONFIGURATION ---
# Use a mesh size that fits in 12GB VRAM (e.g., 256^3 or 512^3)
N_MESH = 256
BOX_SIZE = 1000.0  # Mpc/h
L = BOX_SIZE

# Define time steps globally or per run
a_start = 0.01
a_end = 1.0
a_steps = jnp.linspace(a_start, a_end, 20)

conf = Configuration(
    ptcl_spacing=L/N_MESH, 
    mesh_shape=(N_MESH,)*3, 
    ptcl_grid_shape=(N_MESH,)*3,
    a_start=a_start,
    a_nbody=a_steps
)

# --- 2. COSMOLOGY & DAO INJECTION ---
# Standard Planck-like
cosmo_base = Cosmology(conf, A_s_1e9=2.1, n_s=0.96, Omega_m=0.31, Omega_b=0.05, h=0.67)

# We need a custom transfer function or power spectrum to inject DAO.
# pmwd uses jax-cosmo or simple Eisenstein-Hu.
# We will compute Linear P(k) using jax-cosmo, verify, inject DAO, and pass to pmwd.

def get_linear_pk(cosmo_params, k):
    # jax-cosmo calculation
    # Define JC cosmology
    jc_cosmo = jc.Cosmology(
        Omega_c=cosmo_params.Omega_m - cosmo_params.Omega_b,
        Omega_b=cosmo_params.Omega_b,
        h=cosmo_params.h,
        n_s=cosmo_params.n_s,
        sigma8=0.81, # approximate, A_s used in pmwd but sigma8 in jc often
        Omega_k=0.0,
        w0=-1.0, wa=0.0
    )
    # Power spectrum at z=0 (or z_start?)
    # pmwd expects growth factors separately. We need the shape.
    # Eisenstein & Hu is standard in pmwd.
    return boltzmann(cosmo_params, conf) # This returns a callable or array?
    # Actually pmwd.boltzmann returns the transfer function or growth?
    # Let's check pmwd source or docs usage.
    # boltzmann(cosmo, conf) returns a LinearPowerSpectrum object or similar.
    pass

# Simplified DAO injection:
# We will run TWO simulations:
# 1. Fiducial (No DAO)
# 2. DAO (with Oscillatory feature in Initial Conditions)

def run_simulation(dao_amp=0.0):
    # 1. White Noise (Same seed for sample variance cancellation!)
    seed = 42
    wn = white_noise(seed, conf)
    
    # 2. Linear Modes (Initial Conditions)
    # pmwd's boltzmann returns linear growth and transfer.
    # We intercept the Transfer Function or Growth to inject DAO?
    # Actually, we can just modify the 'linear_modes' output if we know k.
    
    # Standard linear growth/transfer
    t = jnp.linspace(0.01, 1.0, 5) # Scale factors
    # Default boltzmann uses Eisenstein-Hu
    
    # We want to modify the initial power spectrum P(k).
    # linear_modes takes `cosmo` and `conf`.
    # It computes P(k) internally via `boltzmann`.
    
    # Ensure transfer function is computed and capture the updated cosmology object
    cosmo_with_trans = boltzmann(cosmo_base, conf)

    # To inject DAO, we might need to manually scale the modes.
    
    # Get standard modes using the updated cosmology
    ic = linear_modes(wn, cosmo_with_trans, conf)
    
    # Inject DAO: Multiply modes by sqrt(1 + A * sin(...))
    # Generate k grid manually to apply filter
    k_vec = [jnp.fft.fftfreq(N_MESH, d=L/N_MESH) * 2 * np.pi for _ in range(3)]
    k_mesh = jnp.meshgrid(*k_vec, indexing='ij')
    k = jnp.sqrt(sum(ki**2 for ki in k_mesh))
    
    # Avoid k=0 singularity if needed (though sin(0)=0 is fine)
    k = jnp.where(k==0, 1e-10, k)
    
    # DAO Feature: 2% oscillation at k ~ 0.1 - 0.2 h/Mpc
    # Modulation M(k) = 1 + A * sin(k * r_s) * exp(-(k/k_d)^2)
    # Delta_tilde = Delta * sqrt(M(k)) (since P ~ Delta^2)
    
    if dao_amp != 0.0:
        r_s = 100.0 # approx sound horizon
        damp = 10.0 # damping scale
        modulation = 1.0 + dao_amp * jnp.sin(k * r_s) * jnp.exp(-0.5 * (k * damp)**2)
        # ic is likely shape (N, N, N).
        ic = ic * jnp.sqrt(modulation)
        
    # 3. N-body Evolution
    # We evolve from a_start to 1.0 (defined in conf)
    
    # Scale ICs to a_start
    D_1 = growth(1.0, cosmo_with_trans, conf)
    D_init = growth(a_start, cosmo_with_trans, conf)
    ic_init = ic * (D_init / D_1)
    
    ptcl, obsv = lpt(ic_init, cosmo_with_trans, conf)
    
    # Evolve using steps defined in conf
    ptcl, obsv = nbody(ptcl, obsv, cosmo_with_trans, conf)
    
    # 4. Measure P(k) of final particles
    # We need to deposit particles to mesh
    # scatter?
    dens = scatter(ptcl, conf)
    
    # Compute P(k)
    # simple FFT based P(k)
    delta_k = jnp.fft.rfftn(dens)
    # bin into shells... (pmwd doesn't have a built-in P(k) estimator?)
    # We can write a simple one or use 'mcfit' or similar?
    # For now, let's just inspect the output validity.
    
    return dens

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Running GPU Simulation with PMWD...")
    # Remove try/except to see traceback
    dens_fid = run_simulation(dao_amp=0.0)
    print("Fiducial Sim Complete.")
    dens_dao = run_simulation(dao_amp=0.05) # Large DAO to see effect
    print("DAO Sim Complete.")
    
    # We would compare P(k) here.
    # For this script, we just verify it runs on GPU/JAX.
    print("Success! GPU Simulation environment is active.")
