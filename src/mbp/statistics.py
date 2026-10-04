
import torch

def blocked_mean_sem(samples: torch.Tensor, block_size: int):
    """
    Calculates the mean of block averages and the error on the mean
    samples: torch.Tensor(nsamples, nwalkers) - time-ordered observable samples
    block_size: int - samples per block, nsamples % block_size === 0
    return: mean, sem - block mean and error on mean
    """

    nsamples = samples.shape[0]
    if nsamples % block_size != 0:
        raise ValueError(
            "Number of samples, {nsamples}, must be divisible by block_size, {block_size}."
        )

    nblocks = nsamples // block_size
    nwalkers = samples.shape[1]
        
    block_aves = (samples
        .view(nblocks, block_size, nwalkers)
        .mean(dim=1)
        .reshape(-1)) # total independent blocks averages (nblocks * nwalkers)

    mean = block_aves.mean()
    sem = block_aves.std(unbiased=True) / block_aves.numel()**0.5

    return mean, sem

def local_variance_sem(samples: torch.Tensor, block_size: int):
    """
    Calculates variance of local quantities (not on the block averages).
    Calaculates error on the variance using blocked averages of the variance.
    samples: torch.Tensor(nsample, nwalkers) - time-ordered samples
    block_size: int - samples per block
    """

    nsamples = samples.shape[0]
    if nsamples % block_size != 0:
        raise ValueError(
            "Number of samples, {nsamples}, must be divisible by block_size, {block_size}."
        )

    nblocks = nsamples // block_size
    nwalkers = samples.shape[1]
    
    var = samples.reshape(-1).var(unbiased=True) # local variance
    block_vars = (samples
        .view(nblocks, block_size, nwalkers)
        .var(dim=1, unbiased=True)
        .reshape(-1)) # tot number of independent samples of variance (nblocks * nwalkers)
    
    sem = block_vars.std(unbiased=True) / block_vars.numel()**0.5 # standard error on the variance

    return var, sem
