import torch

from .wavefunction import WaveFunction
from .hamiltonian import kinetic_energy, potential_energy

@torch.no_grad()
def metropolis_step(pos, wf, atoms, step_size=0.5):
    """
    pos:       torch.Tensor(batch, nelec, ndim) -current electron positions
    wf:        WaveFunction(nn.Module)          -wavefunction model
    atoms:     torch.Tensor(natoms, ndim)       -nuclear positions
    step_size: float                            -size of cube for random move
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
def thermalize(pos, wf, atoms, n_therm=500, step_size=0.5):
    """
    Run Metropolis burn-in for a single walker.
    pos:       torch.Tensor(batch, nelec, ndim)
    wf:        WaveFunction(nn.Module)
    atoms:     torch.Tensor(natoms, ndim)
    n_therm:   int
    step_size: float
    """
    print(f"    Thermalizing for {n_therm} steps...")

    acc = 0
    for _ in range(n_therm):
        pos, accepted = metropolis_step(pos, wf, atoms, step_size)
        if accepted:
            acc += accepted

    print(f"    Thermalization acceptance rate: {acc / n_therm:.3f}")
    return pos

def estimate_energy(pos, wf, atoms, Z, nsamples=1000, step_size=0.5):
    """
    pos:       torch.Tensor(batch, nelec, ndim)
    wf:        WaveFunction(nn.Module)
    atoms:     torch.Tensor(natoms, ndim)
    Z:         torch.Tensor(natoms,)
    nsamples:  int
    step_size: float
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

def estimate_energy_tensor(pos, wf, atoms, Z, nsamples=1000, step_size=0.5):
    """
    pos:       torch.Tensor(batch, nelec, ndim)
    wf:        WaveFunction(nn.Module)
    atoms:     torch.Tensor(natoms, ndim)
    Z:         torch.Tensor(natoms,)
    nsamples:  int
    step_size: float
    returns: ke, pe torch.Tensor(nsamples, batch)
    """

    print(f"    Sampling for {nsamples} steps...")

    ke = []
    pe = []
    acc = 0

    for i in range(nsamples):
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
    print(f"    Sampling acceptance rate: {accept_rate:.3f}")

    return torch.stack(ke), torch.stack(pe) # (nsamples, batch)

