import torch
import math
from tqdm import tqdm

from .wavefunction import WaveFunction
from .sampler import thermalize, metropolis_step, estimate_energy_tensor
from .hamiltonian import kinetic_energy, kinetic_energy_jvp, potential_energy, Vnn
from .diagnostics import DensityAccumulator
from .sr import sr_update
#from .plotting import plot_density
from .utils import print_section, print_epoch
from .checkpoint import load_checkpoint, save_checkpoint, resolve_state_dict


def run_vmc(
    wf_cfg: WaveFunctionConfig,
    vmc_cfg: VmcConfig,
    atoms: torch.Tensor,
    device: torch.device,
):
    """Runs a variational Monte Carlo optimization.
    For all epochs, the standard errors are calculated using accumulators
    of raw samples to avoid extra memory costs (take caution 
    reporting/relying on this error). A final sampling is performed
    after the final epoch which uses block averages to calculate estimators 
    for a more accurate estimate of the standard error.
    
    Args:

        wf_cfg: Wave function configuration.
        vmc_cfg: VMC sampling and optimization configuration.
        atoms: Atomic positions defining the finite system.
        device: PyTorch device on which to run the calculation.
    """

    # verify_config()
    checkpoint = None
    wf_conflicts = False

    if wf_cfg.checkpoint_load_path:
        wf_cfg_loaded, checkpoint = load_checkpoint(wf_cfg.checkpoint_load_path, device)
        # avoids non-match due to load file name
        wf_cfg_loaded.checkpoint_load_path = wf_cfg.checkpoint_load_path
        if wf_cfg != wf_cfg_loaded:
            wf_conflicts = True
        else:
            wf_cfg = wf_cfg_loaded
        print("Wave function loaded from checkpoint.")

    # print configs
    print_section("Config", "Wavefunction")
    for attr, value in wf_cfg.__dict__.items():
        print(f"   │ {attr}: {value}")
    print_section("Config", "VMC Setup")
    for attr, value in vmc_cfg.__dict__.items():
        print(f"   │ {attr}: {value}")

    print(f"{'─'*62}")

    
    # create Neural Network wavefunction
    wf = WaveFunction(wf_cfg).to(device)
    
    nelec = sum(wf_cfg.n_electrons)
    ndim = wf_cfg.n_dim

    # 'nwalkers' is used synonymously with 'batch'
    nwalkers = vmc_cfg.n_walkers
    # nblock only used to specify independent samples
    nblocks = vmc_cfg.n_blocks
    # mc steps / block
    nsteps = vmc_cfg.n_steps

    # define within a given epoch, since will change
    # should determine block size (~2*tau) emprically by inspecting ACINFO
    #tot_blocks = (nwalkers * nsteps) // nblocks # Neff
    #errfac = 1.0 / math.sqrt(tot_blocks)
    
    if vmc_cfg.seed:
        torch.manual_seed(vmc_cfg.seed)

    # initialize electrons random positions
    pos = torch.randn(nwalkers, nelec, ndim, device=device)
    # convert charge to a tensor
    Z = torch.as_tensor(wf_cfg.charges, device=device)

    # calculate once at start
    Vnn_val = Vnn(atoms, Z)
    
    #if vmc_cfg.epochs <= 1: # verify_config should handle more than one epoch w/ no optimizer chosen
    #    estimate_energy()
    #    return

    lr = vmc_cfg.lr if vmc_cfg.lr is not None else 1e-4

    # initialize wf parameters from checkpoint
    if checkpoint:
        if not wf_conflicts:
            try:
                wf.load_state_dict(checkpoint["model_state"])

            except Exception as e:
                print(f"    ❌ Error loading wave function state: {e}")
        else:
            try:
                print("    Attempting to resolve wave function state...")
                resolve_state_dict(wf, checkpoint["model_state"])
            except Exception as e:
                print(f"    ❌ Error loading wave function state: {e}")
    
    # no reason to not use sr for some parameters and adam/sgd for others no?
    # if parameters frozen
    if vmc_cfg.freeze:
        if 'symmetric_layers' in vmc_cfg.freeze:
            for p in wf.layers.parameters():
                p.requires_grad = False
        if 'ci' in vmc_cfg.freeze:
            for p in wf.det_coefficients.parameters():
                p.requires_grad = False
        if 'jastrow' in vmc_cfg.freeze:
            for p in wf.jastrow_ee.parameters():
                p.requires_grad = False
            for p in wf.jastrow_en.parameters():
                p.requires_grad = False
        if 'orbital' in vmc_cfg.freeze:
            for p in wf.orbitals.orbital_layers.parameters():
                p.requires_grad = False
            for p in wf.orbitals.envelopes.parameters():
                p.requires_grad = False
   
    opt_name = vmc_cfg.optimizer.lower()

    # initialize optimizer with paremeters to update
    
    if opt_name != 'sr':
        if opt_name == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, wf.parameters()), lr=lr)
        elif opt_name == 'adam': # use if sr as well?
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, wf.parameters()), lr=lr)
        else:
            raise ValueError("    Invalid optimization method, choose from 'Adam', 'SGD', or 'SR'.")

        # initialize optimizer state from checkpoint
        if checkpoint:
            if opt_name == checkpoint["optimizer_type"].lower():
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                    print("    Optimizer loaded from checkpoint.")
                    if vmc_cfg.lr:
                        for group in optimizer.param_groups:
                            group['lr'] = lr

                except Exception as e:
                    print(f"    ❌ Error loading optimizer state: {e}")
            else:
                print("    Conflict between checkpoint and VmcConfig optimizer {opt_name}.")
                print("    Optimizer state not loaded.")
    

    '''
    # START OVERRIDE: freeze params and set optimizer, override (TEMP, REMOVE)
    for name, p in wf.named_parameters():
        p.requires_grad_(False)

    #for name, p in wf.named_parameters():
    #    if "jastrow" in name.lower():
    #        p.requires_grad_(True)

    # Freeze all first-order Jastrow terms.
    en_params = list(wf.jastrow_en.parameters())
    ee_params = list(wf.jastrow_ee.parameters())

    # JastrowEN first-order term
    en_params[0].requires_grad_(False)

    # JastrowEE first-order terms
    ee_params[0].requires_grad_(False)
    ee_params[1].requires_grad_(False)

    # Keep higher-order terms trainable
    for p in en_params[1:]:
        p.requires_grad_(True)

    for p in ee_params[2:]:
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [p for p in wf.parameters() if p.requires_grad],
        lr=1e-4,
    )
    print("Optimizing:")
    for p in wf.parameters():
        if p.requires_grad:
            print(f"{p}")
    # end OVERRIDE
    '''
    
    
    # overriding optimizer temporarily
    #optimizer = torch.optim.Adam([
    #    {"params": wf.layers.parameters(), "lr": 3e-5},
    #    {"params": wf.orbitals.orbital_layers.parameters(), "lr": 3e-5},
    #    {"params": wf.orbitals.envelopes.parameters(), "lr": 1e-5},
    #    {"params": wf.jastrow_en.parameters(), "lr": 1e-5},
    #    {"params": wf.jastrow_ee.parameters(), "lr": 1e-5},
    #])


    # PRINT DATA on type of optimizer, which parameters being optimized, and learning rates (and parameters amounts?)
    curr_lr = lr
    if opt_name != 'sr':
        curr_lr = optimizer.param_groups[0]['lr']
        if vmc_cfg.scheduler:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=vmc_cfg.gamma)
            curr_lr = scheduler.get_last_lr()[0]

    print_section("Start", f"VMC Optimization (Epochs {vmc_cfg.epochs:,d})")

    pos = thermalize(pos, wf, atoms, n_therm=vmc_cfg.n_therm, step_size=vmc_cfg.met_step)
    pos.requires_grad_(True)

    # REMOVE_DETS
    #wf.det_mags = torch.zeros_like(wf.det_mags) # zero det mags for next epoch

    print("    Starting optimization loop.")

    # VMC LOOP
    # DENSITY PLOT: checking density along bond axis, make sure even for symm analysis
    #density = DensityAccumulator(zmin=-3, zmax=3, nbins=300)


    # FOR JVP LAPLACIAN
    D = ndim * nelec
    coord_basis = torch.eye(
        D,
        device=device,
        dtype=torch.get_default_dtype(),
    ).reshape(D, nelec, ndim)
    # unrelated to walker batch dim
    laplacian_chunk_size = None


    for epoch in range(vmc_cfg.epochs):

        #print("a0:", wf.jastrow_en.a0.data)
        #print("c_i", wf.det_coefficients)
        
        # for faster gradient construction using per step accumulation
        params = [p for p in wf.parameters() if p.requires_grad]
        # create buffer to store first piece of gradient
        grad_psi = [torch.empty_like(p) for p in params]
        torch._foreach_zero_(grad_psi)
        optimizer.zero_grad(set_to_none=True)

        acc = 0

        # var = sum_i (e_i - e_mean)^2 / (N-1)  = sum_i (e_i^2 + e_mean^2 - 2*e_i*e_mean) / (N-1) = [ sum_i(e_i^2) -2*e_mean*sum_i(e_i) + N*e_mean^2 ] / N-1
        # energy stats
        e = 0.
        k = 0.
        p = 0.
        e2 = 0.
        k2 = 0.
        p2 = 0.
        # accumulate e and e2 over epoch
        # for one epoch: e_mean = e / steps
        #                e_var = [ (1-2*e_mean)*e2 + steps*e_mean^2 ] / (steps-1) ] unbiased=True

        # independent steps = nwalkers * nblocks (per chain)
        # e_std = sqrt(e_var)
        # e_serr = e_std / sqrt(n independent steps)

        # DENSITY PLOT
        #density.reset()

        # SAMPLING LOOP
        # ramp by blocks
        epoch_blocks = int(nblocks*vmc_cfg.block_ramp**epoch)
        epoch_steps = nsteps * epoch_blocks
        tot_samples = epoch_steps * nwalkers
        print_epoch(epoch + 1, vmc_cfg.epochs, epoch_steps, nwalkers)
        errfac = 1.0 / (epoch_blocks * nwalkers)**0.5
        
        if epoch > 0: # per epoch thermalization
            pos = thermalize(
                pos,
                wf,
                atoms,
                n_therm=vmc_cfg.n_therm // 2,
                step_size=vmc_cfg.met_step
            )
            pos.requires_grad_(True)

        # for tqdm
        steps = range(epoch_steps)
        for step in tqdm(
                steps,
                desc="    ┌Sampling",
                unit=" Batch Steps",
                ncols=62,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {rate_fmt}"
        ):
        #for step in range(epoch_steps):
            with torch.no_grad():
                pos, accepted = metropolis_step(
                    pos,
                    wf,
                    atoms,
                    step_size=vmc_cfg.met_step
                )
                if accepted:
                    acc += accepted

            # Evaluate local energy
            pos_sample = pos.detach().requires_grad_(True) # w/o .clone() share storage
            _, logpsi = wf(pos_sample, atoms)             # (nwalkers,)

            # Checking KE/PE
            # pass wf model and store forward pass as a function
            #kin = kinetic_energy_jvp(
            #        wf=wf,
            #        pos=pos_sample,
            #        atoms=atoms,
            #        coord_basis=coord_basis,
            #        chunk_size=laplacian_chunk_size,
            #) # (nwalkers,) 
            kin = kinetic_energy(logpsi, pos_sample)      # (nwalkers,)
            #print(kin-kin_check)
            pot = potential_energy(pos_sample, atoms, Z)   # (nwalkers,)
            E_loc = (kin + pot).detach()                   # (nwalkers,)

            ke = kin.detach()
            pe = pot.detach()

            k += ke.sum().item()                 # (,)
            p += pe.sum().item()                 # (,)
            e += (ke + pe).sum().item()

            k2 += torch.dot(ke, ke).item()
            p2 += torch.dot(pe, pe).item()
            e2 += torch.dot(E_loc, E_loc).item()

            # need two pieces grad(loss) = ave(E_loc * logpsi).backward()) - e_mean*(ave(lopsi.backward) 
            torch.autograd.backward(
                logpsi,
                grad_tensors=E_loc,
                retain_graph=True, # for the next gradient part
            )

            # we now have the grad slot of logpsi storing, p.grad = (E_loc * logpsi).backward()
            # this is storing the grad(logpsi) separately
            grad_psi_step = torch.autograd.grad(
                logpsi,
                params,
                grad_outputs=torch.ones_like(logpsi),
            )

            # grad_psi created earlier
            torch._foreach_add_(grad_psi, grad_psi_step)


        del E_loc, kin, pot, ke, pe

        # stats
        e_mean = e / tot_samples
        k_mean = k / tot_samples
        p_mean = p / tot_samples

        e_var = ( e2 - 2*e_mean*e + tot_samples*e_mean**2 ) / (tot_samples - 1) # this is local energy variance
        k_var = ( k2 - 2*k_mean*k + tot_samples*k_mean**2 ) / (tot_samples - 1)
        p_var = ( p2 - 2*p_mean*p + tot_samples*p_mean**2 ) / (tot_samples - 1)

        e_sem = e_var**0.5 * errfac # these are only exact, sem_ex, if the nblock=2*tau
        k_sem = k_var**0.5 * errfac # if nblock > 2tau, sem > sem_ex 
        p_sem = p_var**0.5 * errfac # if nblock < 2tau, sem < sem_ex
   
        # the above is done to avoid storing two extra nwalker x nblock matrices
        # need per block variance to calculate sem on local energy variance
        # to avoid error prop. calculation of -V/2T (Virial ratio) 
        # calculate mean/sem on block values

        # virial theorem, V = -2T, or R = V / (-2T), if R=1.0 virial satisfied
        # R>1.0 too diffuse, R<1.0 too local
        # only demand R=1 when nuclei are at equilibrium positions
        v_mean = p_mean+Vnn_val
        r_mean = -(v_mean) / 2.0 / k_mean
        cov_kp = (e_var - k_var - p_var) / 2.0
        r_var = (
            p_var / 4.0 / k_mean**2.0 + v_mean**2.0 * k_var / 4.0 / k_mean**4.0 - v_mean / 2.0 / k_mean**3.0 * cov_kp
        )
        r_sem = r_var**0.5 * errfac

        # update parameters
        if opt_name != 'sr':
            scale = 2.0 / tot_samples
            grads = [p.grad for p in params]
            torch._foreach_add_(
                grads,
                grad_psi,
                alpha=-e_mean,
            )
            
            torch._foreach_mul_(
                grads,
                scale,
            )

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(wf.parameters(), max_norm=1.0)

            curr_lr = optimizer.param_groups[0]['lr']
        
        else: # sr
            select_parameters = []
            freeze = vmc_cfg.freeze or []
            if "symmetric_layers" not in freeze:
                select_parameters += list(wf.layers.parameters())
            
            if "ci" not in freeze:
                select_parameters.append(wf.det_coefficients) # nn.Parameter
            
            if "jastrow" not in freeze:
                if wf_cfg.en_poly_order:
                    select_parameters += list(wf.jastrow_en.parameters())
                if wf_cfg.ee_poly_order:
                    select_parameters += list(wf.jastrow_ee.parameters())
            
            if "orbital" not in freeze:
                select_parameters += list(wf.orbitals.parameters())
            
            sr_stats = sr_update(
                wf,
                atoms, 
                positions_flat, 
                energies_flat, 
                select_parameters, 
                curr_lr, 
                diag_shift=1e-2
            )
            print(sr_stats)

        # print/save wf and metrics consistently, step optimizer after printing/saving wf
        if (epoch + 1) % 1 == 0:

            print(f"    │ Local Energy     = {e_mean+Vnn_val:.4f} ± {e_sem:.4f} Ha") 
            print(f"    │ Kinetic Energy   = {k_mean:3.3f} ± {k_sem:.3f} Ha") 
            print(f"    │ Potential Energy = {p_mean+Vnn_val:3.3f} ± {p_sem:.3f} Ha")
            print(f"    │ Virial Ratio     = {r_mean:3.3f} ± {r_sem:.3}")
            print(f"    │ Energy Variance:   {e_var:.3f} ± N/A Ha^2")
            print(f"    │ Acceptance:        {acc / epoch_steps:.3f}")
            #print(f"    │ Loss:              {loss:.6f}")
            print(f"    │ LR:                {curr_lr:.6f}")

            
            metrics = {
                "electronic_energy": e_mean,
                "total_energy": e_mean + Vnn_val,
                "error": e_sem,
                "local_energy_variance": e_var,
                "local_energy_variance_error": 0.,
            }

            if vmc_cfg.checkpoint_save_path:
                path=f"{vmc_cfg.checkpoint_save_path}-{epoch+1}of{vmc_cfg.epochs}"
                save_checkpoint(
                    path,
                    wf,
                    optimizer,
                    wf_cfg,
                    epoch + 1,
                    metrics
                )
                print(f"    Checkpoint saved to {path}.")
            
            del e_sem, e_var, e_mean
            del k_sem, k_var, k_mean
            del p_sem, p_var, p_mean
            del r_sem, r_var, r_mean
            del cov_kp, v_mean
            #del metrics
            # REMOVE_DETS
            #print(f"{wf.det_mags.mean(dim=0)}") # average over batches, (ndets,)
        # REMOVE_DETS
        #wf.det_mags = torch.zeros_like(wf.det_mags) # zero det mags for next epoch

        # DENSITY PLOT
        #rho = density.normalize()
        #plot_density(rho, density.edges, epoch + 1, R=1.4)
        

        if opt_name != 'sr':
            optimizer.step()
            if vmc_cfg.scheduler:
                scheduler.step()
            

        #del rho # DENSITY PLOT
        torch.cuda.empty_cache()

    print(f"\n{'─'*62}")
    
    # Final energy estimate
    print_section("Start", "Final Energy Sampling")
    pos = thermalize(pos, wf, atoms, n_therm=vmc_cfg.n_therm, step_size=vmc_cfg.met_step)
    max_blocks = int(nblocks*vmc_cfg.block_ramp**vmc_cfg.epochs)
    ke_tensor, pe_tensor = estimate_energy_tensor(
        pos,
        wf,
        atoms,
        Z,
        nsamples=nsteps*max_blocks,
        step_size=vmc_cfg.met_step
    )

    energies_tensor = ke_tensor + pe_tensor
    del ke_tensor, pe_tensor

    e_block_means = energies_tensor.view(
        max_blocks,
        nsteps,
        nwalkers
    ).mean(dim=1) #(nblocks, steps/block, nwalkers)
    
    e_means_flat = e_block_means.reshape(-1) # (epoch_blocks * nwalkers) a list of uncorrelated means
    e_mean = e_means_flat.mean()
    e_std = e_means_flat.std(unbiased=True) # mean sample std (internally / nblocks-1)
    
    print(f"    Total samples taken {energies_tensor.numel():,d}")
    print(f"    Independent samples {e_means_flat.numel():,d}")
    
    e_sem = e_std / e_means_flat.numel()**0.5 
    # confidence of mean (68% chance mean is within ±e_sem and 95% chance mean is within ±2e_sem)
    del e_std, e_means_flat, e_block_means

    # we want the variance of the local energy (which is zero when in an eigenstate)
    e_local_var = energies_tensor.reshape(-1).var(unbiased=True) # (samples * nwalkers)
    
    # calculate error bar on the local energy variance estimate
    e_block_vars = energies_tensor.view(
        max_blocks,
        nsteps,
        nwalkers
    ).var(dim=1, unbiased=True) # (epoch_blocks, nwalkers) get variance for each block
    
    e_vars_flat = e_block_vars.reshape(-1) # (epoch_blocks*nwalkers) array of independent variance calcs
    e_var_std = e_vars_flat.std(unbiased=True) # stdev of local energy variance samples
    e_var_sem = e_var_std / e_vars_flat.numel()**0.5 # standard error on local energy variance
    del e_var_std, e_vars_flat, e_block_vars

    # do later
    #ke_block_means = ke_tensor.view(epoch_blocks, nsteps, nwalkers).mean(dim=1) # (epoch_blocks, nwalkers)
    #ke_means_flat = ke_block_means.reshape(-1) # (epoch_blocks*nwalkers)
    #ke_mean = ke_means_flat.mean()
    #ke_std = ke_means_flat.std(unbiased=True)
    #ke_sem = ke_std / ke_means_flat.numel()**0.5
    
    print(f"\n{'─'*62}")
    print_section(" Final Electronic Energy", f"{e_mean.item():.6f} ± {e_sem.item():.6f} Ha")
    # Add Vnn if you want total energy
    #Vnn_val = Vnn(atoms, Z)
    #print(f"Vnn = {Vnn_val:.6f} Ha")
    print_section(" Total Energy", f"{e_mean + Vnn_val:.6f} ± {e_sem:.6f} Ha")
    print(f"{'─'*62}")


    metrics = {
                "electronic_energy": e_mean.item(),
                "total_energy": e_mean + Vnn_val,
                "error": e_sem,
                "local_energy_variance": e_local_var,
                "local_energy_variance_error": e_var_sem,
    }

    if vmc_cfg.checkpoint_save_path:
        save_checkpoint(
            vmc_cfg.checkpoint_save_path,
            wf,
            optimizer,
            wf_cfg,
            vmc_cfg.epochs,
            metrics
        )
        print(f"Checkpoint saved to {vmc_cfg.checkpoint_save_path}.")

