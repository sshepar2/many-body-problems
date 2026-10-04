from typing import Tuple

import math

import torch
from torch import nn

class SymmetricLayer(nn.Module):
    """Single layer of FermiNet with symmetric features and residual connections."""

    def __init__(
        self,
        nspins: Tuple[int, int],
        dim_in_one: int,
        dim_out_one: int,
        dim_in_two: int,
        dim_out_two: int,
        is_last_layer: bool = False,
    ):
        super().__init__()
        self.nspins = nspins
        self.nchannels = sum(1 for n in nspins if n > 0)
        self.is_last_layer = is_last_layer

        # create mask as buffer once and reuse
        #nelec = sum(nspins)
        mask = 1.0 - torch.eye(sum(nspins)) # sum(nspins=nelec)
        self.register_buffer(
            "pair_mask",
            mask.unsqueeze(0).unsqueeze(-1)
        ) # [1, nelec, nelec, 1]

        # Dimension of symmetric features
        dim_g = dim_in_one * (self.nchannels + 1) + dim_in_two * self.nchannels

        # One-electron stream
        self.linear_one = nn.Linear(dim_g, dim_out_one)

        # Two-electron stream (not updated in last layer)
        if not is_last_layer:
            self.linear_two = nn.Linear(dim_in_two, dim_out_two)

    def construct_symmetric_features(
        self,
        h1: torch.Tensor,  # (batch, nelec, ndim)
        h2: torch.Tensor,  # (batch, nelec, nelec, ndim)
    ) -> torch.Tensor:
        """Construct permutation-equivariant features."""

        _, nelec, _ = h1.shape

        nup, ndn = self.nspins

        # Split by spin
        h1_up = h1[:, :nup] if nup > 0 else None # the conditional handles if no spin up
        h1_dn = h1[:, nup:] if ndn > 0 else None # the conditional handles if no spin dn

        # compute once
        h1_up_mean = h1_up.mean(dim=1) if h1_up is not None else None
        h1_dn_mean = h1_dn.mean(dim=1) if h1_dn is not None else None

        # Construct features for each electron
        features = []
        for i in range(nelec):
            feat = [h1[:, i]]  # Own features

            # Mean of h1 over each occupied spin channel
            if h1_up is not None:
                feat.append(h1_up_mean)
            if h1_dn is not None:
                feat.append(h1_dn_mean)

            # Mean of h2[:, i, :] over each occupied spin channel
            if i < nup: # This is a spin-up electron
                if nup > 1: # if there's at least another spin up electron
                    #feat.append(h2[:, i, :nup].mean(dim=1)) # all spin-up electrons (like ave) includes self (change to nup > 0)
                    like_sum = h2[:, i, :nup].sum(dim=1) - h2[:, i, i] # i,i should be zero, but subtracts anyway
                    feat.append(like_sum / (nup - 1)) # uses only non-self-pair features
                else: # if no other spin up electrons
                    feat.append(torch.zeros_like(h2[:, i, i])) # so average is zero

                if ndn > 0:
                    feat.append(h2[:, i, nup:].mean(dim=1)) # all spin-dn electrons (opp ave)
            # i is a spin down electron
            else:
                if ndn > 1: # if there's at least another spin-dn electron
                    #feat.append(h2[:, i, nup:].mean(dim=1)) # all spin-dn electrons (like ave) includes self (change to ndn > 0)
                    like_sum = h2[:, i, nup:].sum(dim=1) - h2[:, i, i] # i,i should be zero, but subtract anyway
                    feat.append(like_sum / (ndn - 1)) # uses only non-self-pair features
                else: # no other spin dn electrons
                    feat.append(torch.zeros_like(h2[:, i, i]))

                if nup > 0:
                    feat.append(h2[:, i, :nup].mean(dim=1)) # all spin-up electrons (opp ave)


            features.append(torch.cat(feat, dim=1))

        return torch.stack(features, dim=1)  # (batch, nelec, dim_g)

    def forward(
        self,
        h1: torch.Tensor,  # (batch, nelec, dim_in_one)
        h2: torch.Tensor,  # (batch, nelec, nelec, dim_in_two)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h1_new: (n_elec, dim_out_one)
            h2_new: (n_elec, n_elec, dim_out_two)
        """
        # Symmetric features
        g = self.construct_symmetric_features(h1, h2)

        # Update one-electron stream with residual
        h1_update = torch.tanh(self.linear_one(g))
        if h1.shape == h1_update.shape:
            h1_new = (h1 + h1_update) / math.sqrt(2.0)
        else:
            h1_new = h1_update

        # Update two-electron stream with residual
        if not self.is_last_layer:
            h2_update = torch.tanh(self.linear_two(h2)) # bias added to the zero diagonals
            if h2.shape == h2_update.shape:
                h2_new = (h2 + h2_update) / math.sqrt(2.0)
            else:
                h2_new = h2_update
            
            # zero diagonals (self-pairs)
            # construct mask before as buffer, nelec doesn't change
            #nelec = h2_new.shape[1]
            #eye = torch.eye(nelec, device=h2_new.device, dtype=h2_new.dtype)
            #mask = 1.0 - eye
            #mask = mask.unsqueeze(0).unsqueeze(-1) # (1, nelec, nelec, 1)
            h2_new = h2_new * self.pair_mask # * mask

        else:
            h2_new = h2  # Not used after this

        return h1_new, h2_new
