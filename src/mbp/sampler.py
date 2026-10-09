import torch
from tqdm import tqdm

from .wavefunction import WaveFunction
from .hamiltonian import kinetic_energy, potential_energy

@torch.no_grad()
def metropolis_step(
    pos: torch.Tensor, # (nwalkers, nelec, ndim), 
    wf: WaveFunction, # (nn.Module)
    atoms: torch.Tensor, # (natoms, ndim),
    step_size: float = 0.5
) -> tuple[torch.Tensor, torch.float]: # (nwalkers, nelec, ndim)
    """Metropolis Step.

    Args:
        pos: Current electron positions.
        wf: Wave function model.
        atoms: Positions of nuclei.
        step_size: Size of cube for random move.
    Returns:
        pos: New electron positions.
        accept.float().mean().item(): Acceptance averaged over walker dimension.
    """

    # Propose move
    pos_new = pos + step_size * torch.randn_like(pos)

    # Evaluate log|ψ|
    _ , logpsi_old = wf(pos, atoms)
    _ , logpsi_new = wf(pos_new, atoms)

    # Acceptance probability
    #log_accept = torch.minimum(torch.tensor(0.0), 2.0 * (logpsi_new - logpsi_old))
    log_accept = 2.0 * (logpsi_new - logpsi_old)

    rand = torch.log(torch.rand_like(log_accept)) # (batch,)

    accept = rand < log_accept # (batch,)

    pos[accept] = pos_new[accept]

    return pos, accept.float().mean().item()


@torch.no_grad()
def thermalize(
    pos: torch.Tensor, # (nwalkers, nelec, ndim)
    wf: WaveFunction, # (nn.Module)
    atoms: torch.Tensor, # (natoms, ndim)
    n_therm: int = 500,
    step_size: float = 0.5
) -> torch.Tensor: # (nwalkers, nelec, ndim)
    """Run Metropolis burn-in for a single walker.
    
    Args:
        pos: Current electron positions.
        wf: Wave function model.
        atoms: Positions of nuclei.
        n_therm: Number of correlated samples to throw away during equilibration
        step_size: Size of cube for random move.
    Returns:
        pos: New electron positions after n_therm samples.   
    """

    print(f"    Thermalizing for {n_therm} steps...")

    acc = 0
    for _ in range(n_therm):
        pos, accepted = metropolis_step(pos, wf, atoms, step_size)
        if accepted:
            acc += accepted

    print(f"    Thermalization acceptance rate: {acc / n_therm:.3f}")
    return pos


def estimate_energy(
    pos: torch.Tensor, # (nwalkers, nelec, ndim)
    wf: WaveFunction, # (nn.Module)
    atoms: torch.Tensor, # (natoms, ndim)
    Z: torch.Tensor, # (natoms,)
    nsamples: int = 1000,
    step_size: float = 0.5
) -> tuple[torch.float, torch.float]:
    """Estimate energy using Metropolis

    Args:
        pos: Current electron positions.
        wf: Wave function model.
        atoms: Positions of nuclei.
        Z: Charges on nuclei, matching the order in atoms tensor.
        nsamples: Number of correlated samples to calculate
        step_size: Size of cube for random move.
    Returns:
        E_mean.mean(): Mean of local energy.
        E_std: Standard deviation of local energy. (Leads to an estimate in sem)
    """

    print(f"    Sampling for {nsamples} steps...")

    E_sum = 0.0
    E2_sum = 0.0
    acc = 0

    for i in range(nsamples):
        with torch.no_grad():
            pos, accepted = metropolis_step(pos, wf, atoms, step_size)
            if accepted:
                acc += accepted

        pos_sample = pos.detach().requires_grad_(True)
        _, logpsi = wf(pos_sample, atoms)

        # leave ndim = 3 for now, also can probably just rewrite the ke part with flattened pos instead
        E_loc = kinetic_energy(logpsi, pos_sample).detach() + potential_energy(pos_sample, atoms, Z).detach()

        E_sum += E_loc
        E2_sum += E_loc**2

    accept_rate = acc / nsamples
    print(f"    Sampling acceptance rate: {accept_rate:.3f}")

    E_mean = E_sum / nsamples
    E_var = E2_sum.mean() / nsamples - E_mean.mean()**2
    E_std = torch.sqrt(E_var) # handle Neff samples outside function, so currently E err not error on E mean

    return E_mean.mean(), E_std


def estimate_energy_tensor(
    pos: torch.Tensor, # (nwalkers, nelec, ndim)
    wf: WaveFunction, # (nn.Module)
    atoms: torch.Tensor, # (natoms, ndim)
    Z: torch.Tensor, # (natoms,)
    nsamples: int = 1000,
    step_size: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.float]: # (nsamples, nwalkers)
    """Estimate energy using Metropolis, retains all sample kinetic and potential energies.

    Args:
        pos: Current electron positions.
        wf: Wave function model.
        atoms: Positions of nuclei.
        Z: Charges on nuclei, matching the order in atoms tensor.
        nsamples: Number of correlated samples to calculate
        step_size: Size of cube for random move.
    Returns:
        torch.stack(ke): Sample and walker resolved kinetic energies.
        torch.stack(pe): Sample and walker resolved potenial energies.
    """

    print(f"    Sampling for {nsamples} steps...")

    ke = []
    pe = []
    acc = 0

    # for tqdm
    samples = range(nsamples)
    for _ in tqdm(
        samples,
        desc="    ┌Sampling",
        unit=" Batch Steps",
        ncols=62,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {rate_fmt}"
    ):
    # for _ in range(nsamples):
        with torch.no_grad():
            pos, accepted = metropolis_step(pos, wf, atoms, step_size)
            if accepted:
                acc += accepted

        pos_sample = pos.detach().requires_grad_(True)
        _, logpsi = wf(pos_sample, atoms)

        # leave ndim = 3 for now, also can probably just rewrite the ke part with flattened pos instead
        ke.append(kinetic_energy(logpsi, pos_sample).detach()) 
        pe.append(potential_energy(pos_sample, atoms, Z).detach())

    accept_rate = acc / nsamples
    print(f"    │ Acceptance rate: {accept_rate:.3f}")

    return torch.stack(ke), torch.stack(pe), accept_rate # (nsamples, batch)

