# Many Body Problems

Real-space variational quantum Monte Carlo using FermiNet-style neural-network wave functions inmplemented in PyTorch.

## Features

- FermiNet-style neural-network wave functions
- Variational Monte Carlo (VMC) optimization
- Metropolis-Hastings sampling with multiple walkers
- Electron-electron and electron-nuclear Jastrow factors
- Envelope functions for orbital/wave function localization
- Configurable neural-network architecture
- Loading/Saving trained wave functions and optimization states
- Example optimization and sampling scripts

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

## Setup

Use `uv` to create a python environment with the required packages.

```bash
uv sync
```

This installs `torch`, `tqdm`, and `matplotlib` by default. 

No example calculations, scripts, or training/sampling functions currently rely on `matplotlib`. It may therefore be removed if desired, but `src/mbp/plotting.py` must also be removed because it explicitly imports the library.

You only need to rerun the command `uv sync` after deleting `.venv` or changing the dependencies in `pyproject.toml`.

## Examples

Example optimization and sampling calculation scripts under `examples/`. Saved wave function states are located in files prefixed with `wf-` and ending in `.pt`. 

Run an optimization with:

```bash
uv run mbp-opt.py
```

Run a sampling calculation with:

```bash
uv run mbp-sample.py
```

These example scripts can be modified to change the physical system, wave function architecture, sampling parameters, and optimization settings.

## Saved Wave Functions

Wave function states are saved as PyTorch `.pt` files. These files contain the information required to reconstruct the wave function as well as optimization settings and calculation metadata.

## Scripts

The helper script `scripts/check.py` can be used to inspect save wave function and calculation state files. In order of increasing detail:

```bash
uv run check.py wf-file.pt
uv run check.py --model wf-file.pt
uv run check.py --model --parameters wf-file.pt
```

The first command displays general information about the saved calculation, including system and optimization settings as raw data of sampled quantities. The `--model` flag adds wave function configuration, including parameter counts, to the output, while `--parameters` displays the wave function parameters values as well.

## Project Structure

many-body-problems
├── examples/       # Example optimizations and sampling calculations
├── scripts/        # Helper and inspection scripts
├── src/mbp/        # Main Python package
├── trained_models/ # Pretrained models with wave function and optimization details 
├── pyproject.toml
└── uv.lock


