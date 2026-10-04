import torch

def kinetic_energy(logpsi, pos):
    """
    logpsi:  torch.Tensor(batch,)
    pos:     torch.Tensor(batch, nelec, ndim)
    returns: torch.Tensor(batch,)
    """

    _, nelec, ndim = pos.shape

    grad_logpsi = torch.autograd.grad(
        logpsi,
        pos,
        grad_outputs=torch.ones_like(logpsi),
        create_graph=True
    )[0] # (batch, nelec, ndim)

    grad_grad_logpsi = torch.zeros_like(logpsi)

    for e in range(nelec):
        for d in range(ndim):
            grad_ed = torch.autograd.grad(
                grad_logpsi[:, e, d].sum(),
                pos,
                create_graph=True
            )[0][:, e, d] # (batch,)
            grad_grad_logpsi += grad_ed
    return -0.5 * (grad_grad_logpsi + (grad_logpsi**2).sum(dim=(1,2)))

def potential_energy(pos, atoms, Z):
    """
    pos:     torch.Tensor(batch, nelec, ndim)
    atoms:   torch.Tensor(natoms, ndim)
    Z:       torch.Tensor(natoms,)
    returns: torch.Tensor(batch,)
    """

    _, nelec, _ = pos.shape
    natoms, _ = atoms.shape # with Z input

    # Ven
    Re = pos.unsqueeze(2)                     # (batch, nelec, 1, ndim)
    Rn = atoms.unsqueeze(0).unsqueeze(0)      # (1, 1, natoms, ndim)
    r_en = torch.norm(Re - Rn, dim=-1)        # (batch, nelec, natom)

    Z_broadcast = Z.view(1, 1, natoms)        # (1, 1, natoms) # with Z input
    Ven = -torch.sum(Z_broadcast / r_en, dim=(1, 2))  # (batch,) with Z input

    # Vee
    Ree = pos.unsqueeze(2) - pos.unsqueeze(1)   # (batch, nelec, nelec, dim)
    r_ee = torch.norm(Ree, dim=-1)              # (batch, nelec, nelec)
    i, j = torch.triu_indices(nelec, nelec, offset=1) # upper triangle indices
    Vee = torch.sum(1.0 / r_ee[:, i, j], dim=1) # (batch,)

    return Vee + Ven # (batch,)

def Vnn(atoms, Z):
    """
    Nuclear-nuclear repulsion
    atoms:   torch.Tensor(natoms, ndim)
    Z:       torch.Tensor(natoms,)
    returns: torch.Tensor(1,)
    """

    natoms, _ = atoms.shape

    if natoms == 1:
        return torch.tensor(0.0)

    Rnn = atoms.unsqueeze(1) - atoms.unsqueeze(0)        # (natoms, natoms, ndim)
    r_nn = torch.norm(Rnn, dim=-1)                       # (natoms, natoms)
    i, j = torch.triu_indices(natoms, natoms, offset=1)  # upper diagonal indices
    return torch.sum(Z[i] * Z[j] / r_nn[i, j])


from .wavefunction import WaveFunction
from torch.func import jvp, vmap

def kinetic_energy_jvp(
    wf: WaveFunction,
    pos: torch.Tensor,         # (nwalkers, nelec, ndim) 
    atoms: torch.Tensor,       # (natoms, ndim)
    coord_basis: torch.Tensor, # (ndim*nelec, nelec, ndim)
    chunk_size=None,
) -> torch.Tensor:             # (nwalkers,)
    
    def logpsi(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, logpsi = wf(x, atoms)

        return logpsi # (nwalkers)


    def first_and_second_directional(direction):
        v = direction.unsqueeze(0).expand_as(pos)


        def first_directional(x):
            _, d1 = jvp(logpsi, (x,), (v,))

            return d1 # (nwalkers,)


        d1, d2 = jvp(first_directional, (pos,), (v,),)

        return d1, d2


    dlogpsi_diag, d2logpsi_diag = vmap(
        first_and_second_directional,
        in_dims=0,
        chunk_size=chunk_size,
    )(coord_basis)

    grad_logpsi_sq = dlogpsi_diag.square().sum(dim=0) # (nwalkers,)
    lap_logpsi = d2logpsi_diag.sum(dim=0)             # (nwalkers,)
    
    return -0.5 * (lap_logpsi + grad_logpsi_sq)       # (nwalkers,)

