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
    
    # He atom
    atoms = torch.tensor(
        [[0., 0., 0.]],
        device=device,
    )

    wf_cfg = WaveFunctionConfig(
        n_electrons=(1, 1),  # 1 up, 1 down
        n_atoms=atoms.shape[0],
        charges=(2,),      # first atom assigned Z=1, 2nd Z=1
        n_determinants=1,
        #en_poly_order=1, # None by default, means no jastrow en term
        ee_poly_order=1, # None by default, means no jastrow ee term
        hidden_dims=((16, 4),),
        bias_orbitals=True,
        #init_orbitals_as_constant=True, # only for start of opt
        use_envelope=True,
        tie_spin_orbitals=True,
        checkpoint_load_path="./he-1-0-1_03.pt"
    )

    vmc_cfg = VmcConfig(
        seed=24,
        n_walkers=64,
        n_steps=40,         # block size
        n_blocks=16,        # total n_samples = n_step * n_blocks
        epochs=5,
        block_ramp=1.025, # n_blocks = prev * (block_ramp) ** epoch
        met_step=0.2,
        n_therm=100,
        optimizer='adam',
        lr=1e-4,
        scheduler=False,
        checkpoint_save_path="./he-1-0-1_04.pt"
    )

    
    run_vmc(wf_cfg, vmc_cfg, atoms, device)
    print("\nFor He atom")
    print("    Exact Total Energy ≈ -2.9037 Ha")


if __name__ == "__main__":
    main()
