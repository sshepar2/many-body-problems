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
from mbp.training_2 import run_vmc
from mbp.utils import print_logo

print_logo(torch.randint(low=1, high=5, size=(1,)))

def main():
    
    # H2 molecule
    R = 1.424
    atoms = torch.tensor(
        [[0., 0., -R / 2.0], [0., 0., R / 2.0]],
        device=device
    )

    wf_cfg = WaveFunctionConfig(
        n_electrons=(1, 1),  # 1 up, 1 down
        n_atoms=atoms.shape[0],
        charges=(1, 1),      # first atom assigned Z=1, 2nd Z=1
        n_determinants=16,
        en_poly_order=2,
        ee_poly_order=2,
        hidden_dims=((128, 16), (128, 16), (128, 16)),
        full_det=False,      # Try False to see difference
        checkpoint_load_path="./wf-h2-16-2-2_03.pt"
    )

    vmc_cfg = VmcConfig(
        n_walkers=64,
        n_steps=40,         # block size
        n_blocks=10,        # total n_samples = n_step * n_blocks
        epochs=5,
        block_ramp=1.025,   # n_blocks = prev * (block_ramp) ** epoch
        met_step=0.3,
        n_therm=100,
        optimizer='adam',
        lr=1e-4,
        checkpoint_save_path="./wf-h2-16-2-2_04.pt"
    )

    
    run_vmc(wf_cfg, vmc_cfg, atoms, device)
    print("\nFor H2 @ R=1.424 bohr")
    print("    Exact Total Energy ≈ -1.1745 Ha")


if __name__ == "__main__":
    main()
