import torch

def autocorr(energies, lag):
    """
    Checks autocorrelation function for batch
    energies: torch.Tensor(nsteps, batch)
    lag: int
    """
    energies = energies - torch.mean(energies, dim=0) # (nsteps, batch) - (1, batch)
    return torch.mean(energies[lag:, :] * energies[:-lag, :], dim=0) / torch.var(energies, dim=0, unbiased=False) # /N (biased) instead of /(N-1)

class DensityAccumulator:
    def __init__(self, zmin=-3.0, zmax=3.0, nbins=200):
        assert nbins % 2 == 0, "Use an even number of bins for mirror symmetry."

        self.zmin = zmin
        self.zmax = zmax
        self.nbins = nbins

        self.hist = None
        self.edges = None

        self.nsamples = 0

        # metrics
        self.left = None
        self.right = None

        self.sum_x = None
        self.sum_x2 = None
        self.sum_x3 = None

    def accumulate(self, positions):
        """
        positions: torch.Tensor(batch, nelec, 3)
        """

        device = positions.device

        if self.hist is None:
            self.hist = torch.zeros(self.nbins, device=device)
            self.edges = torch.linspace(self.zmin, self.zmax, self.nbins+1, device=device)

            self.left = torch.tensor(0.0, device=device)
            self.right = torch.tensor(0.0, device=device)
            self.sum_z = torch.tensor(0.0, device=device)
            self.sum_z2 = torch.tensor(0.0, device=device)
            self.sum_z3 = torch.tensor(0.0, device=device)

        z_coords = positions[:, :, 2].reshape(-1)

        hist = torch.histc(
            z_coords,
            bins=self.nbins,
            min=self.zmin,
            max=self.zmax
        )

        self.hist += hist
        self.nsamples += z_coords.numel()

        # metrics
        self.left += (z_coords < 0.0).sum()
        self.right += (z_coords > 0.0).sum()
        self.sum_z += z_coords.sum()
        self.sum_z2 += (z_coords**2).sum()
        self.sum_z3 += (z_coords**3).sum()

    def normalize(self):

        eps = 1e-12
        mean_z = self.sum_z / self.nsamples
        mean_z2 = self.sum_z2 / self.nsamples
        mean_z3 = self.sum_z3 / self.nsamples
        var_z = mean_z2 - mean_z**2
        std_z = torch.sqrt(var_z.clamp_min(eps))
        third_central = mean_z3 - 3.0 * mean_z * mean_z2 + 2.0 * mean_z**3
        skew_z = third_central / (std_z**3 + eps)
        imbalance = (self.right - self.left) / (self.left + self.right).clamp_min(1.0)
        print(f"Mean z: {mean_z.item():.2f} StDev z: {std_z.item():.3} Skew z: {skew_z.item():.3f} Count Imbalance: {imbalance.item():.2f} Counts: {self.nsamples:,d}")

        dz = (self.zmax - self.zmin) / self.nbins

        if self.nsamples == 0:
            return self.hist

        rho = self.hist / (self.nsamples * dz)
        return rho

    def reset(self):
        if self.hist is not None:
            self.hist.zero_()
            self.left.zero_()
            self.right.zero_()
            self.sum_z.zero_()
            self.sum_z2.zero_()
            self.sum_z3.zero_()
        self.nsamples = 0
