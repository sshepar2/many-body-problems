import torch

import matplotlib.pyplot as plt

def plot_density(rho, edges, epoch, R=1.4):
    """
    rho:   torch.Histc
    edges: torch.Linspace 
    epoch: int
    R:     float
    """

    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure()
    plt.plot(centers.detach().cpu().numpy(),
             rho.detach().cpu().numpy()
    )

    plt.axvline(-R/2, linestyle="--")
    plt.axvline(+R/2, linestyle="--")

    plt.xlabel("z (bond axis)")
    plt.ylabel("ρ(z)")
    plt.title(f"Electron Density – Epoch {epoch}")

    plt.show()
