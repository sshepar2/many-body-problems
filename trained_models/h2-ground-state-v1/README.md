# Overview

This started as a test to get an idea of the computational effort and real time it takes to obtain the exact H₂ ground-state energy on a single A100 GPU. It additionally checks the quality of the wave function, for bugs, and for biases in the code. 

The block size (32) is somewhat larger than $2\tau$, so standard errors are most likely over-estimated (calculated from raw variances, not blocked). Throughput tests were performed on the A100 GPU for the forward pass (no gradients) on this system prior to starting the calculation, and the 65,536 walker test was found to be close to the shoulder of throughput gains. The entire calcUlation uses the Adam optImizer starting with a `1e-4` learning rate and ending with `3e-5` with a total of 830 epochs. See data files for more details on the calculation.

Calculations were submitted via the [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) and performed on a single A100 GPU. 

# File Breakdown

```bash
h2-ground-state-v1/
├── README.md
├── metadata.yaml
├── raw_data
│   ├── energy.pdf 
│   ├── data-plot.p 
│   ├── data-script-per-epoch
│   ├── data-script-per-run
│   ├── data-for-fit
│   └── fit.log
├── mbp.tar
├── run_h2_1-det.py
├── mbp-042.out
└── h2-1-1-1_042.pt
```

The exact mbp package state is provided in `mbp.tar` which includes `pyproject.toml` and the `src/` directory. The package is installed using `python3 -m pip install -e .` in order to retain CUDA versions already installed on the GPU. The information on versions used below can also found at the start of `mbp-042.out`.

| File | Contents |
| ---  | ---      |
| `metadata.yaml` | Metadata about simluated system. |
| `energy.pdf`    | Gnuplot generated PDF of quantities as a function of epoch/opt step |
| `data-plot.p`   | Gnuplot file used to generate PDF and perform fit. |
| `data-script-per-epoch` | Raw space-separated data for each epoch. Used by Gnuplot script |
| `data-script-per-run` | Raw space-separated data for each separate calculation. |
| `data-for-fit` | Raw space-separated data used for fit in Gnuplot script. Contains the final 300 epochs of `data-script-per-epoch` |
| `fit.log` | Gnuplot default output log from fitting procedure |
| `mbp.tar` | Exact state of mbp package used for calculations, includes `pyproject.yaml` and `src/` |
| `run_h2_1-det.py` | Run script used for final calculation. Only save/load files, number of blocks, and learning rate change from calculation to calculation. |
| `mbp-042.out` | Raw output from final calculation. |
| `h2-1-1-1_042.pt` | Checkpoint file of final wave function. |

Since the state of `many-body-problems` code was not committed at the time of this calculate, rather than a commit hash,the exact mbp package state is provided in `mbp.tar` which includes `pyproject.toml` and the `src/` directory. The package is installed using `python3 -m pip install -e .` in order to retain CUDA versions already installed on the GPU. The information on versions used below can also found at the start of `mbp-042.out`.

- Torch: 2.11.0+cu130
- Torch CUDA: 13.0
- GPU: NVIDIA A100-SXM4-40GB
- Compute Capability: (8,0)
- Threads: 6
- Interop Threads: 6
- Tensor Precision: torch.float32

# H₂ Ground State - v1

To parse the checkpoint file `h2-1-1-1_042.pt` use the `check.py` script along with `--model` and `--parameters` for higher level of detail.

## Calculated Quantities

| Quantity | Value |
|---|---:|
| Energy | -1.16772(9) Ha\* |
| ΔE vs CI (-1.1745 Ha) | +6.78 mHa |
| Kinetic energy (T) | 1.226(2) Ha |
| Potential energy (V) | -2.394(2) Ha |
| Virial ratio (R) | 0.976\*\* |
| Local energy variance | 0.044 Ha² |
###### \* SEM actually calculated using block averages.<br>\*\* $R = -V/(2T)$. At equilibrium geometry, require $R=1$.

## Wave Function

- 1 determinants (spin decomposed)
- first order jastrows (electron-nucleus and electron-electron), with a Padé term
- nodeless exponential envelopes on orbitals
- Hidden dimensions: `(128,16) × 3`
- FP32
- Total Parameters: 111,595
  - Linear layers: 111,328
  - Orbtals + Envelopes: 246
  - Jastrow (electron-nucleus): 1
  - Jastrow (electron-electron): 2

## Optimization/Sampling

- 42 calculations run on single A100 GPU
- 830 epochs total
- Optimizer: Adam (no scheduler)
- Learning Rate: 1e-4, 3e-5 (epoch <= 390, epoch > 390)
- 65,536 walkers
- Burn-in: 50, 25 (start of calculation, each epoch)
- Metropolis Step: 0.3 a.u.
- Block size: 32
- Blocks: 2-12 (more blocks as calculations progress)
- Acceptance: 0.66
 
