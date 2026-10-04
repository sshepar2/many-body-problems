import torch
from torch import nn

class JastrowEN(nn.Module):
    def __init__(self, Z, poly_order):
        """
        Z: torch.tensor (natoms,)
        poly_order: int >= 1
        """
        super().__init__()

        assert poly_order >= 1
        self.poly_order = poly_order

        #self.register_buffer("Z", torch.tensor(Z, dtype=torch.float32))

        #natoms = len(Z)

        # learnable pade parameter in denominator
        #self.a0 = nn.Parameter(torch.ones(natoms) * 0.1)
        #self.a0 = nn.Parameter(torch.ones(natoms) * 0.0) # may have been preventing initial decay

        # convert to tensor
        #Z = torch.tensor(Z, dtype=torch.float32)

        # unique Z and mapping back to atom numbers
        Z_types, atom_map = torch.unique(Z, return_inverse=True)

        ntypes = len(Z_types)
        self.register_buffer("Z", Z)                 # (natoms,)
        self.register_buffer("atom_map", atom_map)   # (natoms,)

        self.a0 = nn.Parameter(torch.ones(ntypes) * 0.1)

        # higher order polynomials
        if poly_order >= 2:
            self.a = nn.Parameter(torch.zeros(ntypes, poly_order - 1))
        else:
            self.a = None

    def forward(self, r_ae):
        """
        r_ae:         (batch, nelec, natoms, 1)
        returns: J_en (batch,)
        """

        assert r_ae.shape[2] == self.Z.numel(), (f"r_ae has {r_ae.shape[2]} atoms, but JastrowEN has {self.Z.numel()} charges")

        rij = r_ae.squeeze(-1) # (batch, nelec, natoms)

        a0 = self.a0[self.atom_map]          # (natoms,)

        # Pade cusp term
        pade = (-self.Z * rij) / (1.0 + a0 * rij + 1e-8)

        total = pade

        # higher order polynomials
        if self.poly_order >= 2:
            powers = torch.stack(
                [rij ** (k + 2) for k in range(self.poly_order - 1)],
                dim=-1
            )  # (batch, nelec, natoms, poly_order - 1)

            a = self.a[self.atom_map]       # (atoms, poly_order-1)

            poly = torch.sum(powers * a, dim=-1)

            total = total + poly

        # Sum over electrons and nuclei
        return torch.sum(total, dim=(1, 2)) # (batch,)

class JastrowEE(nn.Module):
    def __init__(self, nup, ndn, poly_order):
        """
        poly_order >= 1
        TODO: e-e cusp correction via analytic laplacian terms, backflow, coordinate transforms
        """
        super().__init__()
        assert poly_order >= 1

        self.poly_order = poly_order

        # fixed constants (cusp conditions)
        self.b0_equ = 1.0 / 4.0
        self.b0_opp = 1.0 / 2.0

        # learnable Pade denominator
        self.b1_equ = nn.Parameter(torch.tensor(0.5))
        self.b1_opp = nn.Parameter(torch.tensor(0.5))

        # learnable higher order coefficients, order 2 and higher
        if poly_order >= 2:
            self.b_equ = nn.Parameter(torch.zeros(poly_order - 1))
            self.b_opp = nn.Parameter(torch.zeros(poly_order - 1))
        else:
            self.b_equ = None
            self.b_opp = None

        # spin masks
        nelec = nup + ndn

        spin = torch.zeros(nelec)
        spin[:nup] = 1.0
        equ_spin = (spin.unsqueeze(0) == spin.unsqueeze(1)).float()
        opp_spin = 1.0 - equ_spin

        eye = torch.eye(nelec)
        equ_spin = equ_spin * (1.0 - eye)
        opp_spin = opp_spin * (1.0 - eye)

        upper = torch.triu(torch.ones(nelec, nelec), diagonal=1)
        equ_spin = equ_spin * upper
        opp_spin = opp_spin * upper

        self.register_buffer("mask_equ", equ_spin)
        self.register_buffer("mask_opp", opp_spin)

    def forward(self, r_ee):
        """
        r_ee: (batch, nelec, nelec, 1)

        returns: J_ee (batch,)
        """

        rij = r_ee.squeeze(-1) # (batch, nelec, nelec)

        # pade term
        pade_equ = (self.b0_equ * rij) / (1.0 + self.b1_equ * rij + 1e-8)
        pade_opp = (self.b0_opp * rij) / (1.0 + self.b1_opp * rij + 1e-8)

        total_equ = pade_equ
        total_opp = pade_opp

        # higher order terms
        if self.poly_order >= 2:
            powers = torch.stack(
                [rij ** (k + 2) for k in range(self.poly_order - 1)],
                dim=-1
            ) # (batch, nelec, nelec, poly_order - 1)

            poly_equ = torch.sum(powers * self.b_equ, dim=-1)
            poly_opp = torch.sum(powers * self.b_opp, dim=-1)

            total_equ = total_equ + poly_equ
            total_opp = total_opp + poly_opp

        # apply masks
        J_equ = torch.sum(total_equ * self.mask_equ, dim=(1, 2))
        J_opp = torch.sum(total_opp * self.mask_opp, dim=(1, 2))

        return J_equ + J_opp # (batch,)
