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
device = torch.device("cpu")
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
from mbp.statistics import blocked_mean_sem, local_variance_sem
from mbp.utils import print_logo

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
        n_steps=40,         # block size
        n_blocks=50,        # total n_samples = n_step * n_blocks
        met_step=0.3,
        n_therm=500,
    )


    # Load wave function config and checkpoint data
    FILE = "./wf-h2-1-1-1_03.pt"
    wf_cfg, checkpoint = load_checkpoint(FILE, device="cpu")

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
    print_section("Energy Sampling", f"Samples: {nsteps * nblocks * nwalkers:,d} (Walkers: {nwalkers:,d})")
    pos = thermalize(pos, wf, atoms, n_therm=vmc_cfg.n_therm, step_size=vmc_cfg.met_step)
    ke_tensor, pe_tensor = estimate_energy_tensor(pos, wf, atoms, Z, nsamples=nsteps*nblocks, step_size=vmc_cfg.met_step)

    ke_mean, ke_sem = blocked_mean_sem(ke_tensor, nblocks)
    pe_mean, pe_sem = blocked_mean_sem(pe_tensor, nblocks)
    e_mean, e_sem = blocked_mean_sem(ke_tensor + pe_tensor, nblocks)
    
    ke_var, ke_var_sem = local_variance_sem(ke_tensor, nblocks)
    pe_var, pe_var_sem = local_variance_sem(pe_tensor, nblocks)
    e_var, e_var_sem = local_variance_sem(ke_tensor + pe_tensor, nblocks)

    del ke_tensor, pe_tensor

    print(f"\n{'─'*62}")
    print_section("Statistics", f"Samples/Block: {nsteps:,d} | Blocks: {nblocks:,d} │ Walkers: {nwalkers:,d}")
    print(f"    ┌Energy Breakdown:")
    print(f"    │ Total Energy:      {e_mean+Vnn_val:3.4f} ± {e_sem:.4f} Ha")
    print(f"    │ Electronic Energy: {e_mean.item():3.4f} ± {e_sem.item():.4f} Ha")
    print(f"    │ Kinetic Energy:     {ke_mean:3.4f} ± {ke_sem:.4f} Ha")
    print(f"    │ Potential Energy:  {pe_mean+Vnn_val:3.4f} ± {pe_sem:.4f} Ha")
    print(f"    │")
    print(f"    │ Energy Variance:    {e_var:.4f} ± {e_var_sem:.4f} Ha^2")
    print(f"    │     KE Variance:    {ke_var:.4f} ± {ke_var_sem:.4f} Ha^2")
    print(f"    │     PE Variance:    {pe_var:.4f} ± {pe_var_sem:.4f} Ha^2")
    print(f"{'─'*62}")


    print("\nFor H2 @ R=1.424 bohr")
    print("    Exact Total Energy ≈ -1.1745 Ha")


if __name__ == "__main__":
    main()
