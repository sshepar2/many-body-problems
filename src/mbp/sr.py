import torch

from tqdm import tqdm

from .wavefunction import WaveFunction


def sr_update(
    wf: WaveFunction,
    atoms: torch.Tensor,
    positions_flat: torch.Tensor,
    energies_flat: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    lr=1e-2,
    diag_shift=1e-3,
    n_sr=None,
    energy_clip_mad=True,
    method='dense' # dense, block, cg (conjugate-gradient). try optimizing using below, but blocking the sublayers of symmetric_layers
):
    """
    Calculate gradients and update parameters using stochastic reconfiguration
    wf: WaveFunction object
    positions_flat: torch.Tensor(nsamples, nelec, ndim) - preferably small SR batch
    energies_flat: torch.Tensor(nsamples,)
    atoms: torch.Tensor(natoms, ndim) - atomic positions
    parameters: list[torch.nn.Parameter]
    lr: float - learning rate
    diag_shift: float - applied to S before taking inverse to avoid singular
    n_sr: int or None - if not None, will randomly select samples from nsamples 
    """

    if len(parameters) == 0:
        raise ValueError("No parameters selected for SR update.")

    # DEBUG
    params_before = [p.detach().clone() for p in parameters]

    device = positions_flat.device
    dtype = positions_flat.dtype
    nsamples = positions_flat.shape[0] # (epoch_steps * nwalkers)

    # Use a small random subset for SR.
    if n_sr and n_sr < nsamples:
        idx = torch.randperm(nsamples, device=device)[:n_sr]
        pos_sr = positions_flat[idx]
        E_sr = energies_flat[idx]
    else:
        pos_sr = positions_flat
        E_sr = energies_flat
    
    E_sr = E_sr.detach()

    # Optional: same idea as your MAD clipping.
    if energy_clip_mad:
        median = torch.median(E_sr)
        mad = torch.median(torch.abs(E_sr - median))
        if mad > 0:
            E_sr = torch.clamp(E_sr, median - 5 * mad, median + 5 * mad)

    O_rows = []

    samples = range(pos_sr.shape[0])
    #for s in range(pos_sr.shape[0]):
    for s in tqdm(
        samples, 
        desc="    ├SR",
        unit=" SR Samples",
        ncols=62,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {rate_fmt}"
    ):
        pos_s = pos_sr[s:s+1].detach()

        _, logpsi_s = wf(pos_s, atoms)
        logpsi_s = logpsi_s.squeeze()

        # these are dlogpsi/dparameter for <psi_i|psi_j> we find this for different R
        grads = torch.autograd.grad(
            logpsi_s,
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        # grads currently in parameters shape (jastrow_en(tensor), jastrow_ee(tensor), orbitals.envelope(), ...)
        grad_flat = []
        for grad, p in zip(grads, parameters):
            if grad is None:
                print("Warning: Found a gradient which is None. Wavefunction does not depend on it.")
                grad_flat.append(torch.zeros_like(p).reshape(-1)) # append (p.numel(),)
            else:
                grad_flat.append(grad.reshape(-1))

        grad_flat = torch.cat(grad_flat) # make torch tensor(p1.numel(), p2.numel,...)
        O_rows.append(grad_flat)
    
    O = torch.stack(O_rows, dim=0)  # (n_sr/nsamples, nparameters): first row: psi_1(R1), psi_2(R1),...
   
    O_centered = O - O.mean(dim=0, keepdim=True) # (n_sr/nsamples, nparameters),  first row: psi_1(R1)-<psi_1>, psi_2(R1)-<psi_2>,...
    E_centered = E_sr - E_sr.mean() # (nsamples,) still column vector now distance from mean

    n = O.shape[0]
    
    g = -2.0 * (O_centered.T @ E_centered) / n # (nparameters, nsamples) * (nsamples,) = (nparameters,)
    # first element (1,): 2*[ (psi_1(R1)-<psi_1>)*E_sr(R1) + (psi_1(R2)-<psi_1>)*E_sr(R2) + ... ] / n 
    
    #if method = 'dense' and len(parameters < 10000):
    
    # if dense, and parameters length isn't too large, just do it
    # if dense, and parameters too large, try sub-matrix problems (so as if method=block)
    # if the parameters are too large in either sub-matrix, then switch to conjugate gradient
    # this logic should be done in the training function and used to set the appropriate 'method' and notify the user
    # don't even let the user decide 'method', if dense is possible, it is fastest. for debugging just hardcode

    S = (O_centered.T @ O_centered) / n # (nparameters, nparameters) 
        # (1,1) element:[ (psi_1(R1)-<psi_1>)*(psi_1(R1)-<psi_1>) + (psi_1(R2)-<psi_1>)*(psi_1(R2)-<psi_1>) + ...] / n
        # (2,1) element:[ (psi_2(R1)-<psi_2>)*(psi_1(R1)-<psi_1>) + (psi_2(R2)-<psi_2>)*(psi_1(R2)-<psi_1>) + ...] / n
        # so this is the matrix of covariances
    
    
        # so g_j = 2/n * sum_i (psi_j(R_i)-<psi_j>)*E_sr(R_i)
        # or g_j = 2 * < E(R)*psi_j(R) - E(R)*<psi_j> >_R
    
        # solve S * delta_p = -g
        # delta_p = -S^{-1}g, but apply diag_shift, and the lr to delta_p globally
    
    eye = torch.eye(S.shape[0], device=device, dtype=dtype)
    
        # cholesky since S in symmetric and positive definite (lowest real is zero, why variance (diag) would most likely be zero? + diag_shift
        #A = S + diag_shift * eye
    L = torch.linalg.cholesky(S + diag_shift * eye)
    dp = torch.cholesky_solve(g[:, None], L).squeeze(1)
        #dp = torch.linalg.solve(S + diag_shift * eye, g)

    idx = 0

    with torch.no_grad():
        for p in parameters:
            np = p.numel()
            step = dp[idx:idx+np].view_as(p)
            #DEBUG
            print("norm relative update:", ((lr*step).norm() / (p.norm()+1e-12)).item())
            print("max |lr*dp|:", (lr*step).abs().max().item())
            p.add_(step, alpha=lr)
            idx += np

    if idx != dp.numel():
        raise RuntimeError(f"Found {idx} parameters to update, but only found {dp.numel()} updates from gradients.")

    # DEBUG
    for i, (p0, p1) in enumerate(zip(params_before, parameters)):
        change = (p1.detach() - p0).abs().max()
        rel = (p1.detach() - p0).norm() / (p0.norm() + 1e-12)

        print(
            i,
            "max change:", change.item(),
            "relative norm change:", rel.item(),
        )
        
    print("predicted_dE (shouldbe negative):", -torch.dot(g, dp * lr))



    return {
        "nsamples": n,
        "nparameters": dp.numel(),
        "g_norm": g.norm().item(),
        "dp_norm": dp.norm().item(),
        "trace_S": torch.trace(S).item(),
    }
