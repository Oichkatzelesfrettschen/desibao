import os
import time
import logging
import traceback
import numpy as np
import matplotlib.pyplot as plt

# Configure JAX for CPU
os.environ["JAX_PLATFORM_NAME"] = "cpu"
# Set device count for pmap/sharding if needed (though pmwd might be single-device by default on CPU)
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=12"

import jax
import jax.numpy as jnp
import pmwd
from pmwd import (
    Configuration,
    Cosmology,
    boltzmann,
    white_noise,
    linear_modes,
    lpt,
    nbody,
    scatter,
    growth,
)

# Setup Logging
logging.basicConfig(
    filename='simulation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_exception(e):
    logging.error(f"Exception occurred: {str(e)}")
    logging.error(traceback.format_exc())
    print(f"Error logged to simulation.log: {str(e)}")

# --- 1. CONFIGURATION ---
N_MESH = 256
BOX_SIZE = 1000.0  # Mpc/h
L = BOX_SIZE

# Define time steps params
a_start = 0.01
a_end = 1.0
n_steps = 20
max_step = (a_end - a_start) / n_steps

# pmwd Configuration
try:
    conf = Configuration(
        ptcl_spacing=L/N_MESH, 
        mesh_shape=(N_MESH,)*3, 
        ptcl_grid_shape=(N_MESH,)*3,
        a_start=a_start,
        a_stop=a_end,
        a_nbody_maxstep=max_step
    )
    logging.info("Configuration initialized successfully.")
except Exception as e:
    log_exception(e)
    # Fallback/Exit if conf fails
    conf = None

# --- 2. COSMOLOGY & SIMULATION ---
cosmo_base = Cosmology(conf, A_s_1e9=2.1, n_s=0.96, Omega_m=0.31, Omega_b=0.05, h=0.67)

def run_simulation(dao_amp=0.0):
    start_time = time.time()
    try:
        logging.info(f"Starting simulation with DAO amplitude: {dao_amp}")
        
        # 1. White Noise
        seed = 42
        wn = white_noise(seed, conf)
        
        # 2. Transfer Function
        cosmo_with_trans = boltzmann(cosmo_base, conf)
        
        # 3. Linear Modes
        ic = linear_modes(wn, cosmo_with_trans, conf)
        
        # 4. Inject DAO
        if dao_amp != 0.0:
            # Match rfftn output shape: (N, N, N//2 + 1)
            kx = jnp.fft.fftfreq(N_MESH, d=L/N_MESH) * 2 * np.pi
            ky = jnp.fft.fftfreq(N_MESH, d=L/N_MESH) * 2 * np.pi
            kz = jnp.fft.rfftfreq(N_MESH, d=L/N_MESH) * 2 * np.pi
            
            k_mesh = jnp.meshgrid(kx, ky, kz, indexing='ij')
            k = jnp.sqrt(sum(ki**2 for ki in k_mesh))
            k = jnp.where(k==0, 1e-10, k)
            
            r_s = 100.0 
            damp = 10.0
            modulation = 1.0 + dao_amp * jnp.sin(k * r_s) * jnp.exp(-0.5 * (k * damp)**2)
            ic = ic * jnp.sqrt(modulation)
            
        # 5. Scale to Start Time (Handled by lpt implicitly via conf.a_start)
        # No manual growth scaling needed if lpt uses conf.a_start
        
        # 6. LPT (Initial Conditions)
        # lpt signature: lpt(modes, cosmo, conf) -> state
        state = lpt(ic, cosmo_with_trans, conf)
        ptcl, obsv = state
        
        # 7. N-Body Evolution
        # Corrected call: nbody(ptcl, obsv, cosmo, conf)
        # Time steps defined in conf
        state = nbody(ptcl, obsv, cosmo_with_trans, conf)
        
        ptcl, obsv = state
        
        # 8. Measure Density
        dens = scatter(ptcl, conf)
        
        duration = time.time() - start_time
        logging.info(f"Simulation completed in {duration:.2f} seconds.")
        return dens, duration
        
    except Exception as e:
        log_exception(e)
        return None, time.time() - start_time

if __name__ == "__main__":
    print("Running Simulation on CPU (12 cores)...")
    if conf is None:
        print("Configuration failed. Exiting.")
        exit(1)
        
    stats = []
    
    # Run Fiducial
    print("Running Fiducial Simulation...")
    dens_fid, t_fid = run_simulation(dao_amp=0.0)
    if dens_fid is not None:
        jnp.save("artifacts/data/sim_results/dens_fid.npy", dens_fid)
    stats.append(f"Fiducial: {t_fid:.2f}s")
    
    # Run DAO
    print("Running DAO Simulation...")
    dens_dao, t_dao = run_simulation(dao_amp=0.05)
    if dens_dao is not None:
        jnp.save("artifacts/data/sim_results/dens_dao.npy", dens_dao)
    stats.append(f"DAO (5%): {t_dao:.2f}s")
    
    # Report
    report_path = "artifacts/reports/simulation_performance.txt"
    with open(report_path, "w") as f:
        f.write("Performance Report\n")
        f.write("==================\n")
        for s in stats:
            f.write(s + "\n")
            
    print("Done. Check simulation.log and simulation_performance.txt.")
