from typing import Tuple, Sequence
from dataclasses import dataclass

@dataclass
class WaveFunctionConfig:
    """Configuration for a wave function.
    
    Attributes:
        n_electrons: Number of each electron spin type (n_up, n_down)
        n_atoms: Number of atoms
        charges: Charges on each nuclei
        n_dim: Number of real-space dimensions of physical system
        n_determinants: Number of detereminants used in wave function
        full_det: Maintain full Slater matrix (as opposed to spin-separated) when calculating determinants
        en_poly_order: Order of the polynomial expansion to include in electron-nuclear Jastrow
        ee_poly_order: Order of the polynomial expansion to include in electron-electron Jastrow
        hidden_dims: Shape and depth of linear layers of the neural network
        bias_orbital: Add a bias parameter to the output of the orbital layers
        init_orbitals_as_constant: Initially sets the output of orbital layer to all ones
        use_envelope: Apply an envelope function to the orbitals
        tie_spin_orbitals: Use only one orbital head per pair of up/down electrons (closed-shell)
        checkpoint_load_path: Path to .pt file to be loaded for calculation
    """
    # System
    n_electrons: Tuple[int, int]
    n_atoms: int                 # remove, add atom positions determine len from charges/atoms
    charges: Sequence[int]
    n_dim: int = 3
    
    # Determinants
    n_determinants: int = 1
    full_det: bool = False
    
    # Jastrow
    en_poly_order: int = None   # combine into tuple (en, ee) remove one
    ee_poly_order: int = None
    
    # Orbital
    hidden_dims: Tuple[Tuple[int, int], ...] = ((256, 32), (256, 32), (256, 32))
    bias_orbitals: bool = False       
    init_orbitals_as_constant: bool = False
    use_envelope: bool = True
    tie_spin_orbitals: bool = False
    #jastrow_freeze: bool = False
    
    checkpoint_load_path: str = None

@dataclass
class VmcConfig:
    """Configuration for variational Monte Carlo optimization
    
    Attributes:
        
        seed: Seed to fix random number generator start for initial samples
        n_walkers: Number of walkers to use in the VMC calculation
        n_steps: Number of samples in a block
        n_blocks: Number of blocks to sample per epoch
        epochs: Number of epochs (optimization steps) for the optimization
        block_ramp: Number controlling how n_blocks increases from epoch to epoch:
            int(n_blocks*block_ramp**epoch)
        met_step: Step size to take during Metropolis samping
        n_therm: Number of samples to reach equilibration (applies initially and half before each epoch)
        freeze: Specifies which parameters in the wave function to freeze during optimization:
            'symmetric_layers', 'ci', 'orbital', 'jastrow' 
        optimizer: Optimizer to use in calculation: 'adam', 'sgd', 'sr', 'min_sr'
        lr: Learning rate
        scheduler: Use torch.optim.lr_scheduler.ExponentialLR scheduler
        gamma: Gamma value for the scheduler
        print_autocorr: Performs and prints autocorrelation analysis during calculation
        checkpoint_save_path: Path to '.pt' file to save the wave function and results of the calculation
    """

    # Sampling
    seed: int = None
    n_walkers: int = 1
    n_steps: int = 50
    n_blocks: int = 20
    epochs: int = 10
    block_ramp: float = 1.0
    met_step: float = 0.5
    n_therm: int = 200

    # Optimizer
    freeze: Sequence[str] = None     # move to OptimizerConfig
    optimizer: str = 'adam'
    lr: float = None
    scheduler: bool = False 
    gamma: float = 1.0

    # Printing
    print_autocorr: bool = False     # move to PrintConfig

    checkpoint_save_path: str = None

