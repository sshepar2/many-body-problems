import torch
import math
from tqdm import tqdm

from .wavefunction import WaveFunction
from .sampler import thermalize, metropolis_step, estimate_energy_tensor
from .hamiltonian import kinetic_energy, potential_energy, Vnn
from .diagnostics import DensityAccumulator
from .sr import sr_update
from .plotting import plot_density
from .utils import print_section, print_epoch
from .checkpoint import load_checkpoint, save_checkpoint, resolve_state_dict


# MUST SORT OUT

def run_vmc(
    wf_cfg,
    vmc_cfg,
    atoms,
    device,
    optimizer=None,
    scheduler=None,
):

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
                print(f"❌ Error loading wave function state: {e}")
        else:
            try:
                print("Attempting to resolve wave function state")
                resolve_state_dict(wf, checkpoint["model_state"])
            except Exception as e:
                print(f"❌ Error loading wave function state: {e}")
    
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
            raise ValueError("Invalid optimization method, choose from 'Adam', 'SGD', or 'SR'.")

        # initialize optimizer state from checkpoint
        if checkpoint:
            if opt_name == checkpoint["optimizer_type"].lower():
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                    print("Optimizer loaded from checkpoint.")
                    if vmc_cfg.lr:
                        for group in optimizer.param_groups:
                            group['lr'] = lr

                except Exception as e:
                    print(f"❌ Error loading optimizer state: {e}")
            else:
                print("Conflict between checkpoint and VmcConfig optimizer {opt_name}.")
                print("Optimizer state not loaded.")
    

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

    for epoch in range(vmc_cfg.epochs):

        #print("a0:", wf.jastrow_en.a0.data)
        #print("c_i", wf.det_coefficients)
        positions = []
        energies = []
        acc = 0

        # TEMP: Checking KE/PE
        ke = []
        pe = []

        # DENSITY PLOT
        #density.reset()

        # SAMPLING LOOP
        # ramp by blocks
        epoch_blocks = int(nblocks*vmc_cfg.block_ramp**epoch)
        epoch_steps = nsteps * epoch_blocks
        print_epoch(epoch + 1, vmc_cfg.epochs, epoch_steps, nwalkers)
        errfac = 1.0 / epoch_blocks**0.5
        
        if epoch > 0: # per epoch thermalization
            pos = thermalize(pos, wf, atoms, n_therm=vmc_cfg.n_therm // 2, step_size=vmc_cfg.met_step)
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
                pos, accepted = metropolis_step(pos, wf, atoms, step_size=vmc_cfg.met_step)
                if accepted:
                    acc += accepted

            # Evaluate local energy
            pos_sample = pos.detach().requires_grad_(True) # w/o .clone() share storage
            _, logpsi = wf(pos_sample, atoms)              # (nwalkers,)

            # TEMP: Checking KE/PE
            kin = kinetic_energy(logpsi, pos_sample)       # (nwalkers,)
            pot = potential_energy(pos_sample, atoms, Z)   # (nwalkers,)
            E_loc = kin + pot                              # (nwalkers,)

            ke.append(kin.detach())                        # (...epoch_steps, nwalkers)
            pe.append(pot.detach())                        # (...epoch_steps, nwalkers)

            # DENSITY PLOT
            #density.accumulate(pos.detach())

            # Store positions to call forward on later, energies need for average, no grad
            positions.append(pos.detach())                 # (...epoch_steps, nwalkers, nelec, ndim)
            energies.append(E_loc.detach())                # (...epoch_steps, nwalkers)

            if vmc_cfg.print_autocorr:
                switch_off=False
                if (step + 1) % (epoch_steps//2) == 0 and not switch_off: # every 5,000 steps (printing 100 lines is alot)
                    switch_off = True # only do once
                    # Move everything below into a function
                    print(f"\nStep {step+1} of {new_steps}\n")
                    print_section("Calculating", "Autocorrelation Info")
                    print("Lag    Rho(k)    Tau")
                    tau = torch.tensor(0.5)
                    for k in range(1, 101):
                        rho = autocorr(energies, k) # (batch,)
                        rho = torch.mean(rho)
                        if rho < 1e-2: # handles sign switch as well, since will start positive
                            print(f"For step_size {vmc.met_step}, could safely truncate at k={k-1}")
                            print(f"Using a block size of {(torch.ceil(2*tau) // 1).item()}")
                            print(f"Dividing final error by sqrt(N_eff) = ({(torch.sqrt(new_steps//torch.ceil(2*tau))).item()})")
                        tau += rho
                        print(f"{k:3d}    {rho:2.3f}    {tau.item():2.3f}")

        del E_loc, kin, pot

        # organize for efficient calculation of logpsi values again w/o graphs for positions
        positions_tensor = torch.stack(positions)                                 # (epoch_steps, nwalkers, nelec, ndim)
        del positions
        positions_flat = positions_tensor.view(epoch_steps * nwalkers, nelec, ndim) # (epoch_steps * nwalkers, nelec, ndim)

        energies_tensor = torch.stack(energies)                      # (epoch_steps, nwalkers)
        del energies
        energies_flat = energies_tensor.view(epoch_steps * nwalkers) # (epoch_steps * nwalkers,)
        
        if opt_name != 'sr':
            optimizer.zero_grad()
            #loss = torch.zeros(())
            #for pos_sample, E_sample in zip(positions, energies):
            #    pos_sample = pos_sample.requires_grad_(True)
            _, logpsi_all = wf(positions_flat, atoms)
            #    loss += (E_sample - E_mean) * logpsi


            # clipping step  # DELETE IF NO IMPROVEMENT, OR CAUSES PROBLEMS
            median = torch.median(energies_flat)
            diff = torch.abs(energies_flat - median)
            mad = torch.median(diff)
            minE = median - 5 * mad
            maxE = median + 5 * mad
            energies_flat = torch.clamp(energies_flat, minE, maxE)
            ##################

            loss = torch.mean((energies_flat - energies_flat.mean()) * logpsi_all)
            #loss = 2 * loss / (len(positions) * nwalkers)
            loss.backward() # derivatives w.r.t. parameters

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
            # starts as (nsteps, nwalkers)
            e_block_means = energies_tensor.view(epoch_blocks, nsteps, nwalkers).mean(dim=1) #(nblocks, steps/block, nwalkers)
            #e_block_means = e_blocks.mean(dim=1) # (epoch_blocks, nwalkers) all independent mean estimates
            e_means_flat = e_block_means.reshape(-1) # (epoch_blocks * nwalkers) a list of uncorrelated means
            e_mean = e_means_flat.mean()
            e_std = e_means_flat.std(unbiased=True) # mean sample std (internally / nblocks-1)
            e_sem = e_std / e_means_flat.numel()**0.5 # confidence of mean (68% chance mean is within ±e_sem and 95% chance mean is within ±2e_sem)

            # ke mean and sem
            ke_tensor = torch.stack(ke) # (epoch_steps, nwalkers)
            ke_block_means = ke_tensor.view(epoch_blocks, nsteps, nwalkers).mean(dim=1) # (epoch_blocks, nwalkers)
            ke_means_flat = ke_block_means.reshape(-1) # (epoch_blocks*nwalkers)
            ke_mean = ke_means_flat.mean()
            ke_std = ke_means_flat.std(unbiased=True)
            ke_sem = ke_std / ke_means_flat.numel()**0.5

            #pe mean and sem
            pe_tensor = torch.stack(pe) # (epoch_steps, nwalkers)
            pe_block_means = pe_tensor.view(epoch_blocks, nsteps, nwalkers).mean(dim=1) # (epoch_blocks, nwalkers)
            pe_means_flat = pe_block_means.reshape(-1) # (epoch_blocks*nwalkers)
            pe_mean = pe_means_flat.mean()
            pe_std = pe_means_flat.std(unbiased=True)
            pe_sem = pe_std / pe_means_flat.numel()**0.5

            # we want the variance of the local energy (which is zero when in an eigenstate)
            e_local_var = energies_flat.var(unbiased=True) # (epoch_steps * nwalkers)
            # calculate error bar on the local energy variance estimate
            e_block_vars = energies_tensor.view(epoch_blocks, nsteps, nwalkers).var(dim=1, unbiased=True) # (epoch_blocks, nwalkers) get variance for each block
            e_vars_flat = e_block_vars.reshape(-1) # (epoch_blocks*nwalkers) array of independent variance calcs
            e_var_std = e_vars_flat.std(unbiased=True) # stdev of local energy variance samples
            e_var_sem = e_var_std / e_vars_flat.numel()**0.5 # standard error on local energy variance


            print(f"    │ Local Energy     = {e_mean+Vnn_val:.4f} ± {e_sem:.4f} Ha") 
            print(f"    │ Kinetic Energy   = {ke_mean:3.3f} ± {ke_sem:.3f} Ha") 
            print(f"    │ Potential Energy = {pe_mean+Vnn_val:3.3f} ± {pe_sem:.3f} Ha")
            print(f"    │ Energy Variance:   {e_local_var:.3f} ± {e_var_sem:.3f} Ha^2")
            print(f"    │ Acceptance:        {acc / epoch_steps:.3f}")
            #print(f"    │ Loss:              {loss:.6f}")
            print(f"    │ LR:                {curr_lr:.6f}")

            
            metrics = {
                "electronic_energy": e_mean.item(),
                "total_energy": e_mean + Vnn_val,
                "error": e_sem,
                "local_energy_variance": e_local_var,
                "local_energy_variance_error": e_var_sem,
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
            
            del e_sem, e_std, e_mean, e_means_flat, e_block_means
            del ke_sem, ke_std, ke_mean, ke_means_flat, ke_block_means, ke_tensor
            del pe_sem, pe_std, pe_mean, pe_means_flat, pe_block_means, pe_tensor
            del e_var_sem, e_var_std, e_vars_flat, e_block_vars, e_local_var 
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
            
            del loss
            del maxE, minE, mad, diff, median
            del logpsi_all
            
        

        #del rho # DENSITY PLOT
        
        del energies_flat, energies_tensor
        del ke, pe
        del positions_flat, positions_tensor
        torch.cuda.empty_cache()

    print(f"\n{'─'*62}")
    
    # Final energy estimate
    print_section("Start", "Final Energy Sampling")
    pos = thermalize(pos, wf, atoms, n_therm=vmc_cfg.n_therm, step_size=vmc_cfg.met_step)
    max_blocks = int(nblocks*vmc_cfg.block_ramp**vmc_cfg.epochs)
    ke_tensor, pe_tensor = estimate_energy_tensor(pos, wf, atoms, Z, nsamples=nsteps*max_blocks, step_size=vmc_cfg.met_step)

    energies_tensor = ke_tensor + pe_tensor
    del ke_tensor, pe_tensor

    e_block_means = energies_tensor.view(max_blocks, nsteps, nwalkers).mean(dim=1) #(nblocks, steps/block, nwalkers)
    e_means_flat = e_block_means.reshape(-1) # (epoch_blocks * nwalkers) a list of uncorrelated means
    e_mean = e_means_flat.mean()
    e_std = e_means_flat.std(unbiased=True) # mean sample std (internally / nblocks-1)
    e_sem = e_std / e_means_flat.numel()**0.5 # confidence of mean (68% chance mean is within ±e_sem and 95% chance mean is within ±2e_sem)
    del e_std, e_means_flat, e_block_means

    # we want the variance of the local energy (which is zero when in an eigenstate)
    print("total samples used to calculate variance", energies_tensor.numel())
    e_local_var = energies_tensor.reshape(-1).var(unbiased=True) # (samples * nwalkers)
    # calculate error bar on the local energy variance estimate
    e_block_vars = energies_tensor.view(max_blocks, nsteps, nwalkers).var(dim=1, unbiased=True) # (epoch_blocks, nwalkers) get variance for each block
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

