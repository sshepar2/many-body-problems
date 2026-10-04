# Many Body Problems

Real-space quantum Monte Carlo nearal wave function

## Features

- 

## Requirements

- uv
- Python 3.10+

## Setup

Use the uv command below to create a python environment with the necessary packages.

```bash
uv sync
```

The above command will install `torch`, `tqdm`, and `matplotlib` by default. No example calculations, scripts, or training/sampling functions currently rely on `matplotlib` so this can be removed if desired but in order for the `mbp` library to build correctly, the file `src/mbp/plotting.py` must be removed first, which is the only file which explicitly imports the `matplotlib` library.

You only need to rerun the command above if you delete the `.venv` or change the dependencies in `pyproject.toml`.

## Examples

There are example optimization and sampling calculation scripts under `examples/` with saved states (prefixed with `wf-`). These can be run using,

```bash
uv run mbp-opt.py
```

and

```bash
uv run mbp-sample.py
```

## Scripts

In order to get more information about the state files which contain data on the calculation and wave function parameters there is a helper script located at `scripts/check.py`, with usage (in order of increasing details)

```bash
uv run check.py wf-file.pt
uv run check.py --model wf-file.pt
uv run check.py --model --parameters wf-file.pt
```

