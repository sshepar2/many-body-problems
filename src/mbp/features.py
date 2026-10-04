from typing import Tuple

import torch
from torch import nn


class InputFeatures(nn.Module):
    """Construct geometric features from positions."""

    def __init__(self, natoms: int, ndim: int = 3):
        super().__init__()

    def forward(
        self,
        pos: torch.Tensor,      # (batch, nelec, ndim,)
        atoms: torch.Tensor,    # (natoms, ndim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            ae:   (batch, nelec, natoms, ndim) - electron-atom vectors
            r_ae: (batch, nelec, natoms, 1)    - electron-atom distances
            ee:   (batch, nelec, nelec, ndim)  - electron-electron vectors
            r_ee: (batch, nelec, nelec, 1)     - electron-electron distances
        """
        _, nelec, _ = pos.shape

        # Electron-atom features

        # (batch, nelec, natoms, ndim)
        ae = pos.unsqueeze(2) - atoms.unsqueeze(0).unsqueeze(0)
        # (batch, nelec, natoms, 1)
        r_ae = torch.norm(ae, dim=3, keepdim=True)

        # Electron-electron features
        ee = pos.unsqueeze(2) - pos.unsqueeze(1)   # (batch, nelec, nelec, ndim)
        # Mask diagonal to avoid division by zero in gradients
        eye = torch.eye(nelec, device=pos.device)
        eye = eye.unsqueeze(0).unsqueeze(-1)              # (1, nelec, nelec, 1)
        # (1, nelec, nelec, 1)
        r_ee = torch.norm(ee + eye, dim=3, keepdim=True) * (1.0 - eye)

        return ae, r_ae, ee, r_ee


class FeatureLayer(nn.Module):
    """Create initial features for one- and two-electron streams."""

    def __init__(self, natoms: int, ndim: int = 3):
        super().__init__()
        # Output dimensions
        self.one_electron_features = natoms * (ndim + 1)  # distance+vector/atom
        self.two_electron_features = ndim + 1             # distance + vector

    def forward(
        self,
        ae: torch.Tensor,    # (batch, nelec, natoms, ndim)
        r_ae: torch.Tensor,  # (batch, nelec, natoms, 1)
        ee: torch.Tensor,    # (batch, nelec, nelec, ndim)
        r_ee: torch.Tensor,  # (batch, nelec, nelec, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h1: (nelec, natoms * (ndim+1)) - one-electron features
            h2: (nelec, nelec, ndim+1) - two-electron features
        """

        batch, nelec, _, _ = ae.shape

        # One-electron: concatenate distances and vectors, then flatten
        ae_features = torch.cat([r_ae, ae], dim=3) # (batch, nelec, natoms, ndim+1)
        h1 = ae_features.reshape(batch, nelec, -1) # (batch, nelec, natoms*(ndim+1))

        # Two-electron: concatenate distances and vectors
        h2 = torch.cat([r_ee, ee], dim=3)  # (batch, nelec, nelec, ndim+1)

        return h1, h2
