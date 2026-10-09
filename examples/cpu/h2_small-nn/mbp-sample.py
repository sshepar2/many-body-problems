import torch

print(f"{'─'*62}")
print("Torch / CUDA Information")
print(f"{'─'*62}")

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print("Memory allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("Memory reserved:", torch.cuda.memory_reserved() / 1024**2, "MB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Force CPU
device = torch.device("cpu") # remove line for gpu
print("Using device (forced):", device)
print("Threads:", torch.get_num_threads())
print("Interop Threads:", torch.get_num_interop_threads())
print("torch.Tensor precision:", torch.get_default_dtype())

from mbp.config import WaveFunctionConfig, VmcConfig
from mbp.wavefunction import WaveFunction
from mbp.hamiltonian import Vnn
from mbp.checkpoint import load_checkpoint
from mbp.sampler import thermalize, estimate_energy_tensor
from mbp.utils import print_section
from mbp.statistics import blocked_mean_sem, local_variance_sem, blocked_covariance
from mbp.utils import print_logo
from mbp.diagnostics import autocorr

print_logo(torch.randint(low=1, high=5, size=(1,)))

def main():
   
    # H2 molecule
    R = 1.424
    atoms = torch.tensor(
        [[0., 0., -R / 2], [0., 0., R / 2]],
        device=device,
    )

    # just a convenient structure, won't pass directly anywhere
    vmc_cfg = VmcConfig(
        n_walkers=64,
        n_steps=40,          # block size
        n_blocks=50,         # total n_samples = n_step * n_blocks
        met_step=0.3,        # adjust until acceptance is near 0.6
        n_therm=500,
        print_autocorr=True, # performs/displays autocorrelation analysis
    )


    # Load wave function config and checkpoint data
    FILE = "./wf-h2-1-1-1_03.pt"
    wf_cfg, checkpoint = load_checkpoint(FILE, device="cpu")

    print("\n┌Wave function loaded from checkpoint.\n") # should be in the above function

    wf = WaveFunction(wf_cfg).to(device)
    wf.load_state_dict(checkpoint["model_state"])
    
    # shorter defs
    nelec = sum(wf_cfg.n_electrons)
    ndim = wf_cfg.n_dim
    nwalkers = vmc_cfg.n_walkers
    nsteps = vmc_cfg.n_steps
    nblocks = vmc_cfg.n_blocks

    # simple seed
    seed = 42
    torch.manual_seed(seed)

    # initialize electrons random positions
    pos = torch.randn(nwalkers, nelec, ndim, device=device, dtype=torch.float32)
    # convert charge to a tensor
    Z = torch.as_tensor(wf_cfg.charges, device=device, dtype=torch.float32)

    # calculate once at start
    Vnn_val = Vnn(atoms, Z)

    # Final energy estimate
    print_section(
        "Energy Sampling",
        f"Samples: {nsteps * nblocks * nwalkers:,d} (Walkers: {nwalkers:,d})"
    )
    pos = thermalize(
        pos,
        wf,
        atoms,
        n_therm=vmc_cfg.n_therm,
        step_size=vmc_cfg.met_step
    )
    ke_tensor, pe_tensor, acc = estimate_energy_tensor(
        pos,
        wf,
        atoms,
        Z,
        nsamples=nsteps*nblocks,
        step_size=vmc_cfg.met_step
    )

    ke_mean, ke_sem = blocked_mean_sem(ke_tensor, nblocks)
    pe_mean, pe_sem = blocked_mean_sem(pe_tensor, nblocks)
    e_mean, e_sem = blocked_mean_sem(ke_tensor + pe_tensor, nblocks)
    
    ke_var, ke_var_sem = local_variance_sem(ke_tensor, nblocks)
    pe_var, pe_var_sem = local_variance_sem(pe_tensor, nblocks)
    e_var, e_var_sem = local_variance_sem(ke_tensor + pe_tensor, nblocks)

    # virial ratio and sem, R = -V / (2T)
    v_mean = pe_mean + Vnn_val
    r_mean = -v_mean / 2.0 / ke_mean

    # needed for R_sem
    cov = blocked_covariance(ke_tensor, pe_tensor, nblocks)
    ke_blocked_var = ke_sem**2.0 * nblocks * nwalkers
    pe_blocked_var = pe_sem**2.0 * nblocks * nwalkers
    
    # delta method variance
    r_blocked_var = (
        v_mean**2.0 * ke_blocked_var / (4.0 * ke_mean**4.0)
        + pe_blocked_var / (4.0 * ke_mean**2.0)
        - v_mean * cov / (2.0 * ke_mean**3.0)
    ) / nblocks / nwalkers

    r_sem = torch.sqrt(r_blocked_var)
    
    print(f"\n{'─'*62}")
    print_section("Statistics", f"Samples/Block: {nsteps:,d} │ Blocks: {nblocks:,d} │ Walkers: {nwalkers:,d}")
    print(f"    ┌Energy Breakdown:")
    print(f"    │ Total Energy:      {e_mean+Vnn_val:8.4f} ± {e_sem:.4f} Ha")
    print(f"    │ Electronic Energy: {e_mean:8.4f} ± {e_sem:.4f} Ha")
    print(f"    │ Kinetic Energy:    {ke_mean:8.4f} ± {ke_sem:.4f} Ha")
    print(f"    │ Potential Energy:  {v_mean:8.4f} ± {pe_sem:.4f} Ha")
    print(f"    │ Virial Ratio (R):  {r_mean:8.4f} ± {r_sem:.4f} Ha")
    print(f"    │")
    print(f"    │ Energy Variance:   {e_var:8.4f} ± {e_var_sem:.4f} Ha²")
    print(f"    │     KE Variance:   {ke_var:8.4f} ± {ke_var_sem:.4f} Ha²")
    print(f"    │     PE Variance:   {pe_var:8.4f} ± {pe_var_sem:.4f} Ha²")
    print(f"{'─'*62}")

    print("\n┌For H2 @ R=1.424 bohr")
    print("│   Exact Total Energy ≈ -1.1745 Ha")

    if vmc_cfg.print_autocorr:
        tau, k = autocorr(ke_tensor + pe_tensor)

        block_size = int(torch.ceil(2.0 * tau))
        print(f"\n{'─'*62}")
        print_section("Diagnostic Results", f"Corr. Samples: {nsteps * nblocks:,d} │ Walkers: {nwalkers:,d}")
        print(f"    ┌Autocorr. Breakdown:")
        print(f"    │ Total Corr. Samples:    {nsteps * nblocks:,d}")
        print(f"    │ Metropolis Step:        {vmc_cfg.met_step:.2f} a.u.")
        print(f"    │ Acceptance:             {acc:.2f}")
        print(f"    │ Max Lag (k):            {k:,d}")
        print(f"    │ Tau:                    {tau.item():2.3f}")
        print(f"    │ Block Size (2xTau):     {block_size:,d}")
        print(f"    │ Averaged Walkers:       {nwalkers:,d}")
        print(f"{'─'*62}")

        if not 0.3 < acc < 0.7:
            print("\nWarning: Advice Ahead.\n")
            print(f"┌Your acceptance value ({acc:.2f}) does not usually result in")
            print("│   efficient sampling for most systems. A weak constraint:")
            print("│   0.2 < acc. < 0.7, or stronger one: 0.3 < acc. < 0.6")
            print("│   generally results in the optimal sampling.\n")
            print(f"┌Consider varying met_step and repeating this diagnostic")
            print("│   aiming to either make the acceptance value fall within")
            print("│   the ranges mentioned, or to confirm your met_step") 
            print("│   leads to the minimum block size (2xTau).\n")
            print("┌Increasing (Decreasing) met_step will decrease (increase)")
            print("│   acceptance but can increase or decrease the correlation")
            print("│   time/block size (Tau/2xTau).\n")
            print(f"{'─'*62}")
        print("\nFinal Recommendation.\n")
        if acc >= 0.7:
            print(f"┌Your acceptance is a little high (>= 0.7).")
            print(f"│   For better efficiency (a smaller 2xTau) try increasing")
            print("│   met_step.")
        if acc <= 0.3:
            print(f"┌Your acceptance is a little low (<= 0.3).")
            print(f"│   For better efficiency (a smaller 2xTau) try decreasing")
            print("│   met_step.")
        print(f"\n┌With a Metropolis step size of {vmc_cfg.met_step:.2f} (acceptance {acc:.2f}), the")
        print(f"│   appropriate block size for this system is at least: {block_size:,d}.")
        print(f"\n┌Note 1: Do not rely on a single diagnostic result. Repeat")
        print("│   calculations using different random seeds.")
        print(f"\n┌Note 2: Optimal met_step and block size (n_step) may vary as")
        print("│   wave functions optimize.\n")
        print(f"{'─'*62}")


    del ke_tensor, pe_tensor


if __name__ == "__main__":
    main()
    
    print("\nCalculation Complete.\n")

