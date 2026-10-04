from typing import List, Tuple

import torch
from torch import nn
from torch.profiler import record_function

from .config import WaveFunctionConfig
from .features import InputFeatures, FeatureLayer
from .layers import SymmetricLayer
from .jastrow import JastrowEN, JastrowEE
from .orbitals import Orbitals


class WaveFunction(nn.Module):
    """
    Top-level wave function.
    Returns sign and log|Ψ| for a configuration.
    """

    def __init__(self, config: WaveFunctionConfig):
        super().__init__()
        self.config = config
        Z = torch.as_tensor(config.charges)

        # Input feature construction
        self.input_features = InputFeatures(config.n_atoms, config.n_dim)
        self.feature_layer = FeatureLayer(config.n_atoms, config.n_dim)

        # Layers
        self.layers = nn.ModuleList()

        dim_one = self.feature_layer.one_electron_features
        dim_two = self.feature_layer.two_electron_features

        nspins = config.n_electrons

        for i, (h1_dim, h2_dim) in enumerate(config.hidden_dims):
            is_last = (i == len(config.hidden_dims) - 1)
            layer = SymmetricLayer(
                nspins=nspins,
                dim_in_one=dim_one,
                dim_out_one=h1_dim,
                dim_in_two=dim_two,
                dim_out_two=h2_dim,
                is_last_layer=is_last,
            )
            self.layers.append(layer)
            dim_one = h1_dim
            dim_two = h2_dim

        # Orbital construction
        self.orbitals = Orbitals(config)

        # Jastrow construction
        self.use_jastrow_en = True if config.en_poly_order else False
        self.use_jastrow_ee = True if config.ee_poly_order else False
        
        if self.use_jastrow_en:
            self.jastrow_en = JastrowEN(Z, config.en_poly_order)
        if self.use_jastrow_ee:
            self.jastrow_ee = JastrowEE(nspins[0], nspins[1], config.ee_poly_order)

        # Determinant Coefficients, for easier global optimization
        self.det_coefficients = nn.Parameter(torch.ones(config.n_determinants) / config.n_determinants) # (ndet,)

    def forward(
        self,
        pos: torch.Tensor,     # (batch, nelec, ndim,)
        atoms: torch.Tensor,   # (natoms, ndim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            sign: scalar, sign of Ψ
            log_abs_psi: scalar, log|Ψ|
        """
        # added some precautions
        # atoms defined outside config, so at mercy of separate definition
        atoms = atoms.to(device=pos.device, dtype=pos.dtype)
        assert atoms.shape[0] == self.config.n_atoms, (
            f"atoms has {atoms.shape[0]} atoms, but config.n_atoms = {self.config.n_atoms}"
        )

        # Construct geometric features
        with record_function("WF::input_features"):
            ae, r_ae, ee, r_ee = self.input_features(pos, atoms)

        # Initial features
        with record_function("WF::feature_layers"):
            h1, h2 = self.feature_layer(ae, r_ae, ee, r_ee)

        # Pass through FermiNet layers
        with record_function("WF::layers"):
            for i, layer in enumerate(self.layers): # added only for profiling
                #for layer in self.layers: # only keep this line
                with record_function(f"WF::layer_{i}"):
                    h1, h2 = layer(h1, h2) # and this line when profiling removed

        # Construct orbitals with envelope
        with record_function("WF::orbitals"):
            orbital_matrices = self.orbitals(h1, r_ae)

        # Compute determinants
        with record_function("WF::_compute_determinants"):
            signs, log_dets = self._compute_determinants(orbital_matrices)

        # construct jastrows
        with record_function("WF::jastrow_en"):
            jastrow_en = self.jastrow_en(r_ae) if self.use_jastrow_en else 0.
        with record_function("WF::jastrow_ee"):
            jastrow_ee = self.jastrow_ee(r_ee) if self.use_jastrow_ee else 0.

        # Sum over determinants using logsumexp
        with record_function("WF::_Logsumexp_signed"):
            sign, log_abs_sum = self._logsumexp_signed(signs, log_dets)

        return sign, log_abs_sum + jastrow_en + jastrow_ee

    def _compute_determinants(
        self,
        orbital_matrices: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sign and log|det| for each determinant.

        Args:
            orbital_matrices: List of (batch, ndets, nelec, norb) tensors

        Returns:
            signs: (batch, ndets,)
            log_abs_dets: (batch, ndets,)
        """
        # single det
        if len(orbital_matrices) == 1:
            # Full determinant case: single (batch ndets, nelec, nelec) matrix
            matrix = orbital_matrices[0] # (batch, ndets, nelec, nelec)
            # CHECK THIS, orbital_matrices is returned in [] array, so [0] unpacks this
            #print(matrix.shape) # should be (batch, ndets, nelec, nelec)
            signs, log_abs_dets = torch.linalg.slogdet(matrix) # (batch, ndets)
        else:
            # Spin-factored case: product of determinants
            all_signs = []
            all_log_dets = []
            for matrix in orbital_matrices:
                sign, log_det = torch.linalg.slogdet(matrix)
                all_signs.append(sign)
                all_log_dets.append(log_det)

            # Product of signs, sum of log determinants
            signs = torch.stack(all_signs).prod(dim=0)
            log_abs_dets = torch.stack(all_log_dets).sum(dim=0)

        return signs, log_abs_dets

    def _logsumexp_signed(
        self,
        signs: torch.Tensor,       # (batch, ndets,)
        log_values: torch.Tensor,  # (batch, ndets,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sign and log of sum of signed values in log space.

        Returns sum = sign * exp(log_abs_sum) given:
            sum_i sign_i * exp(log_values_i)
        """
        # Compute max for numerical stability
        max_log = log_values.max(dim=1, keepdim=True).values

        signed_sum = (signs * self.det_coefficients * torch.exp(log_values - max_log)).sum(dim=1) # (1,)

        # Extract sign and log absolute value
        sign = torch.sign(signed_sum)
        log_abs_sum = torch.log(torch.abs(signed_sum) + 1e-20) + max_log.squeeze(1)

        return sign, log_abs_sum
