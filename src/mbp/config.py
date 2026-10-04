from typing import Tuple, Sequence
from dataclasses import dataclass

@dataclass
class WaveFunctionConfig:
    """Configuration for the wave function."""
    # System
    n_electrons: Tuple[int, int]     # (n_up, n_down)
    n_atoms: int                     # remove and determine from len of charges
    charges: Sequence[int]           # nuclear charges
    n_dim: int = 3
    
    # Determinants
    n_determinants: int = 1
    full_det: bool = False
    
    # Jastrow
    en_poly_order: int = None           # combine into tuple (en, ee) remove one
    ee_poly_order: int = None
    
    # Orbital
    hidden_dims: Tuple[Tuple[int, int], ...] = ((256, 32), (256, 32), (256, 32))
    bias_orbitals: bool = False
    init_orbitals_as_constant: bool = False # sets output of orbital layers to 1 initially.
    use_envelope: bool = True        # false if debugging or doing hydrogen (or single atom+single electron)
    tie_spin_orbitals: bool = False  # will attempt closed-shell calculation, i.e. one orbital head, use for both spins
    #jastrow_freeze: bool = False
    
    # Reuse
    checkpoint_load_path: str = None # load wf/optimizer from file (if path given, will try to initialize params)

@dataclass
class VmcConfig:
    """Configuration for variational Monte Carlo optimization"""
    seed: int = None
    n_walkers: int = 1
    n_steps: int = 50                # block size
    n_blocks: int = 20               # total n_samples = n_step * n_blocks
    epochs: int = 10
    block_ramp: float = 1.0         # n_samples = prev * (sample_rmap) ** epoch
    met_step: float = 0.5
    n_therm: int = 1000
    freeze: Sequence[str] = None     # 'symmetric_layers', 'ci', 'orbital', 'jastrow' 
    optimizer: str = 'adam'          # 'adam', 'sgd', 'sr'
    lr: float = None                #        
    scheduler: bool = False          # remove when put in OptimizerConfig
    gamma: float = 1.0               # scheduler learning rate decay, remove when put in OptimizerConfig
    print_autocorr: bool = False     # remove once in PrintConfig
    checkpoint_save_path: str = None # file to safe wf/optimizer

