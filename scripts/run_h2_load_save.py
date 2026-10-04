import torch


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

print("Using device:", device)

print(torch.get_num_threads())
print(torch.get_num_interop_threads())


from mbp.config import WaveFunctionConfig, VmcConfig
from mbp.training import run_vmc

def main():
    
    R = 1.424

    # later use function to import xyz file or something
    atoms = torch.tensor(
        [[0., 0., -R / 2], [0., 0., R / 2]],
        device=device,
        dtype=torch.float32
    )  # Two protons

    # H2 molecule
    wf_cfg = WaveFunctionConfig(
        n_electrons=(1, 1),  # 1 up, 1 down
        n_atoms=atoms.shape[0],
        charges=(1, 1),      # first atom assigned Z=1, 2nd Z=1
        n_determinants=16,
        en_poly_order=2,
        ee_poly_order=2,
        hidden_dims=((128, 16), (128, 16), (128, 16)),
        jastrow_freeze=False, # freeze all jastrow params, cusp conditions imposed
        full_det=False,      # Try False to see difference
        checkpoint_load_path="./checkpoints/h2-16-2-2_17.pt",       # must match the wf structure defined here
    )

    vmc_cfg = VmcConfig(
        n_walkers=64,
        n_steps=40,         # block size
        n_blocks=50,        # total n_samples = n_step * n_blocks
        epochs=5,
        block_ramp=1.025,  # n_blocks = prev * (block_ramp) ** epoch
        met_step=0.4,
        n_therm=500,
        lr=None,
        scheduler=True,
        gamma=0.95,
        print_autocorr=False,
        checkpoint_save_path="./checkpoints/h2-16-2-2_18_2.pt",
    )

    
    run_vmc(wf_cfg, vmc_cfg, atoms, device)
    print("\nFor H2 @ R=1.424 bohr")
    print("    Exact Total Energy ≈ -1.174 Ha")


if __name__ == "__main__":
    main()
