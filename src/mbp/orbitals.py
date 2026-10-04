from typing import List, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .config import WaveFunctionConfig

class Orbitals(nn.Module):
    """
    Transform one-electron features into orbitals and apply envelope.
    Handles both full_det and spin-factored cases.
    """

    def __init__(self, config: WaveFunctionConfig):
        super().__init__()
        self.config = config
        self.nup, self.ndn = config.n_electrons
        self.nelec = self.nup + self.ndn
        self.natoms = config.n_atoms

        # Determine dimensions from last hidden layer
        self.dim_in = config.hidden_dims[-1][0]  # One-electron stream output

        # Orbital projection layers (one per occupied spin channel)
        self.orbital_layers = nn.ModuleList()

        self.envelopes = nn.ModuleList()

        active_spins = [s for s in config.n_electrons if s > 0]

        # appears to be unused after populated here
        #self.norbitals = []

        self.shared_spin_orbitals = (
            config.tie_spin_orbitals
            and not config.full_det
            and self.nup == self.ndn
            and self.nup > 0
        )
        
        if config.tie_spin_orbitals and not self.shared_spin_orbitals:
            raise Warning(
                "Could not tie spin orbitals, requires, nup == ndn > 0, and full_det=False."
                + "\nFound: nup: {self.nup}, ndn: {self.ndn}, full_det={config.full_det}."
            )

        if config.init_orbitals_as_constant and not config.bias_orbitals:
            raise Warning(
                "Setting WaveFunctionConfig.bias_orbitals=True to initialize constant orbitals."
            )
            config.bias_orbitals=True


        if self.shared_spin_orbitals:
            print("    Sharing orbitals between spin channels.")
            # only need one set of spin orbitals
            norbitals = self.nup *  config.n_determinants
            
            layer = nn.Linear(self.dim_in, norbitals, bias=config.bias_orbitals)

            if config.init_orbitals_as_constant:
                with torch.no_grad():
                    layer.weight.zero_()
                    layer.bias.fill_(1.0)

            self.orbital_layers.append(layer)

            if config.use_envelope:
                self.envelopes.append(
                    ExponentialEnvelopeNodeless(
                        norbitals=norbitals,
                        natoms=config.n_atoms,
                        Z=config.charges,
                    )
                )
            else:
                self.envelopes.append(NoEnvelope())


        else: # unrestricted
            for nspin in active_spins:
                if config.full_det:
                    # Need n_total_electrons * n_dets orbitals per electron
                    norbitals = self.nelec * config.n_determinants
                else:
                    # Need n_spin * n_dets orbitals per electron
                    norbitals = nspin * config.n_determinants

                layer = nn.Linear(self.dim_in, norbitals, bias=config.bias_orbitals)

                if config.init_orbitals_as_constant:
                    with torch.no_grad():
                        layer.weight.zero_()
                        layer.bias.fill_(1.0)

                self.orbital_layers.append(layer)


                #self.orbital_layers.append(
                #    nn.Linear(self.dim_in, norbitals, bias=config.bias_orbitals)
                #)

                # apply envelope
                if config.use_envelope:
                    #self.envelopes.append(
                    #    ExponentialEnvelope(
                    #        norbitals=norbitals,
                    #        natoms=config.n_atoms,
                    #        init_alpha_raw=-2.0,
                    #    )
                    #)
                    self.envelopes.append(
                        ExponentialEnvelopeNodeless(
                            norbitals=norbitals,
                            natoms=config.n_atoms,
                            Z=config.charges,
                        )
                    )
                else:
                    self.envelopes.append(NoEnvelope())

                # does not appear to be utilized here or elsewhere
                #self.norbitals.append(norbitals)
            
    def forward(
        self,
        h1: torch.Tensor,    # (batch, nelec, dim_in)
        r_ae: torch.Tensor,  # (batch, nelec, natoms, 1)
    ) -> List[torch.Tensor]:
        """
        Returns:
            List of orbital matrices, one per spin channel (or combined if full_det).
            Each shape: (n_dets, n_elec_channel, n_orb_channel)
        """

        batch, _, _ = h1.shape

        r_ae = r_ae.squeeze(-1)  # (batch, nelec, natoms)

        h1_channels = []
        r_channels = []
        active_spins = []

        if self.nup > 0:
            h1_channels.append(h1[:, :self.nup])
            r_channels.append(r_ae[:, :self.nup])
            active_spins.append(self.nup)

        if self.ndn > 0:
            h1_channels.append(h1[:, self.nup:])
            r_channels.append(r_ae[:, self.nup:])
            active_spins.append(self.ndn)

        # Project to orbitals
        orbitals = []

        if self.shared_spin_orbitals:
            # single shared spatial orbital map for both spin channels
            # already checked nup==ndn and full_det=False
            layer = self.orbital_layers[0] # only a single layer at this point
            envelope = self.envelopes[0]   # only a single envelope at this point
            
            for h1_spin, r_spin in zip(h1_channels, r_channels):
                orb = layer(h1_spin)        # (batch, nspin, norbitals)
                orb = envelope(orb, r_spin) # envelope or identity
                orbitals.append(orb)        
        else:    

            for h1_spin, r_spin, layer, envelope in zip(
                h1_channels,
                r_channels,
                self.orbital_layers,
                self.envelopes,
            ):
                orb = layer(h1_spin)         # (batch, nspin, norbitals)
                orb = envelope(orb, r_spin)  # envelope or identity
                orbitals.append(orb)


        # Reshape into determinant matrices
        shaped_orbitals = []
        # moved above
        #active_spins = [s for s in self.config.n_electrons if s > 0]

        for orb, nspin in zip(orbitals, active_spins):
            # Shape: full det (batch, nspin, ndets * nelec) OR (batch, nspin, ndets * nspin)
            # Reshape to: full det (batch, nspin, ndets, nelec) OR (batch, nspin, ndets, nspin)
            #assert orb.shape[-1] == nspin * self.config.n_determinants
            #orb = orb.view(batch, nspin, self.config.n_determinants, -1)
            # try reshape instead of view
            orb = orb.reshape(batch, nspin, self.config.n_determinants, -1)
            # Transpose to: full det (batch, ndets, nspin, nelec) OR (batch, ndets, nspin, nspin)
            # last dim can also be thought of as norbitals_per_spin
            orb = orb.permute(0, 2, 1, 3)

            shaped_orbitals.append(orb)

        # If full_det, concatenate spin channels along electron dimension
        if self.config.full_det:
            # Concatenate: (batch, ndets, nup, nelec) and (batch, ndets, ndown, nelec)
            # -> (batch, ndets, nelec, nelec)
            shaped_orbitals = [torch.cat(shaped_orbitals, dim=2)]

        return shaped_orbitals


class NoEnvelope(nn.Module):
    """Identity envelope: leaves orbital outputs unchanged."""

    def forward(
        self,
        orb: torch.Tensor,      # (batch, nspin, norbitals)
        r_spin: torch.Tensor,   # (batch, nspin, natoms)
    ) -> torch.Tensor:
        return orb


class ExponentialEnvelope(nn.Module):
    """
    Atom-centered exponential orbital envelope.

    envelope_{ik} = sum_A c_{kA} exp(-alpha_{kA} r_{iA})
    """

    def __init__(
        self,
        norbitals: int,
        natoms: int,
        init_alpha_raw: float = -2.0,
    ):
        super().__init__()

        self.raw_alpha = nn.Parameter(
            torch.full((norbitals, natoms), init_alpha_raw)
        )

        self.coeff = nn.Parameter(
            torch.ones(norbitals, natoms) / natoms
        )

    def forward(
        self,
        orb: torch.Tensor,      # (batch, nspin, norbitals)
        r_spin: torch.Tensor,   # (batch, nspin, natoms)
    ) -> torch.Tensor:

        alpha = F.softplus(self.raw_alpha) + 1e-6  # (norbitals, natoms)

        exp_terms = torch.exp(
            -r_spin.unsqueeze(2) * alpha.unsqueeze(0).unsqueeze(0)
        )  # (batch, nspin, norbitals, natoms)

        envelope = torch.sum(
            exp_terms * self.coeff.unsqueeze(0).unsqueeze(0),
            dim=-1,
        )  # (batch, nspin, norbitals)

        return orb * envelope


class ExponentialEnvelopeNodeless(nn.Module):
    """
    Atom-centered nodeless exponential orbital envelope.
    Useful for ground-states, simple bonding orbital systems
    envelope_{ik} = sum_A c_{kA} exp(-alpha_{kA} r_{iA})
    """

    def __init__(
        self,
        norbitals: int,
        natoms: int,
        Z: Sequence[int], # (natoms,)
        #alpha_init: float = 1.0,
    ):
        super().__init__()

        alpha_init = torch.as_tensor(Z, dtype=torch.get_default_dtype()) # (natoms,)
        # use for helium
        #alpha_init = torch.as_tensor(1.6875, dtype=torch.get_default_dtype()) # (natoms,)
        alpha_raw = torch.log(torch.expm1(alpha_init)) # (natoms,)

        self.alpha_raw = nn.Parameter(
            alpha_raw.unsqueeze(0).expand(norbitals, natoms).clone() # (norbitals, natoms)
        )

        self.coeff_raw = nn.Parameter(
            torch.zeros(norbitals, natoms)
        )

    def forward(
        self,
        orb: torch.Tensor,      # (batch, nspin, norbitals)
        r_spin: torch.Tensor,   # (batch, nspin, natoms)
    ) -> torch.Tensor:

        alpha = F.softplus(self.alpha_raw) + 1e-6  # (norbitals, natoms)
        coeff = F.softmax(self.coeff_raw, dim=-1)  # (norbitals,) initially 1.0

        exp_terms = torch.exp(
            -r_spin.unsqueeze(2) * alpha.unsqueeze(0).unsqueeze(0)
        )  # (batch, nspin, norbitals, natoms)

        envelope = torch.sum(
            exp_terms * coeff.unsqueeze(0).unsqueeze(0),
            dim=-1,
        )  # (batch, nspin, norbitals)

        return orb * envelope
        #return envelope
