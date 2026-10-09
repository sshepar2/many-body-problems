import torch


def blocked_mean_sem(
    samples: torch.Tensor, # (nsamples, nwalkers)
    block_size: int
) -> tuple[torch.Tensor, torch.Tensor]: # [(1,), (1,)]
    """
    Calculates the mean of block averages and the error on the mean
    samples: torch.Tensor(nsamples, nwalkers) - time-ordered observable samples
    block_size: int - samples per block, nsamples % block_size === 0
    return: mean, sem - block mean and error on mean
    """

    nsamples, nwalkers = samples.shape
    if nsamples % block_size != 0:
        raise ValueError(
            "Number of samples, {nsamples}, must be divisible by block_size, {block_size}."
        )

    nblocks = nsamples // block_size
        
    block_aves = (
        samples
        .reshape(nblocks, block_size, nwalkers) # attempts `.view`
        .mean(dim=1)
        .reshape(-1)
    ) # total independent blocks averages (nblocks * nwalkers)

    mean = block_aves.mean()
    sem = block_aves.std(unbiased=True) / block_aves.numel()**0.5

    return mean, sem # to get var of blocked averages: sem**2 * nblocks * nwalkers

def local_variance_sem(
    samples: torch.Tensor, # (nsamples, nwalkers)
    block_size: int
) -> tuple[torch.Tensor, torch.Tensor]: # [(1,), (1,)]
    """
    Calculates variance of local quantities (not on the block averages).
    Calaculates error on the variance using blocked averages of the variance.
    samples: torch.Tensor(nsample, nwalkers) - time-ordered samples
    block_size: int - samples per block
    """

    nsamples, nwalkers = samples.shape
    if nsamples % block_size != 0:
        raise ValueError(
            "Number of samples, {nsamples}, must be divisible by block_size, {block_size}."
        )

    nblocks = nsamples // block_size
    
    var = samples.reshape(-1).var(unbiased=True) # local variance
    block_vars = (
        samples
        .reshape(nblocks, block_size, nwalkers) # attempts to do `.view`
        .var(dim=1, unbiased=True)
        .reshape(-1)
    ) # tot number of independent samples of variance (nblocks * nwalkers)
    
    sem = block_vars.std(unbiased=True) / block_vars.numel()**0.5 # standard error on the variance

    return var, sem

def blocked_covariance(
    samples_a: torch.Tensor, # (nsamples, nwalkers)
    samples_b: torch.Tensor, # (nsamples, nwalkers)
    block_size: int
) -> torch.Tensor: # (1,)
    """Calculates the covariance between two quantities
    using blocked averages

    Args:
        samples_a: Time-ordered raw sampled values of quantity a.
        samples_b: Time-ordered raw sampled values of quantity b.
        block_size: Number of consecutive samples to average for s block sample.

    Returns:
        cov_ab: The covariance of the block averages of quantity a and b.
    """
    nsamples, nwalkers = samples_a.shape
    if nsamples % block_size != 0:
        raise ValueError(
            "Number of samples, {nsamples}, must be divisible by block_size, {block_size}."
        )

    if nsamples != samples_b.shape[0] or nwalkers != samples_b.shape[1]:
        raise ValueError(
            "nsamples, nwalkers must be the same in samples_a and samples_b."
        )
    
    nblocks = nsamples // block_size
        
    block_aves_a = (
        samples_a
        .reshape(nblocks, block_size, nwalkers) # note, if possible this attempts `.view`
        .mean(dim=1)
        .reshape(-1)
    ) # total independent blocks averages (nblocks * nwalkers)

    block_aves_b = (
        samples_b
        .reshape(nblocks, block_size, nwalkers)
        .mean(dim=1)
        .reshape(-1)
    ) # (nblocks * nwalkers) in same order as block_aves_a, so they match up as sampled.

    mean_a = block_aves_a.mean()
    mean_b = block_aves_b.mean()

    #var_a = block_aves_a.var(unbiased=True)
    #var_b = block_aves_b.var(unbiased=True)

    cov_ab = (
        (block_aves_a - mean_a) * (block_aves_b - mean_b)
    ).sum() / (nblocks * nwalkers - 1)
    
    #sem = block_aves.std(unbiased=True) / block_aves.numel()**0.5

    return cov_ab
