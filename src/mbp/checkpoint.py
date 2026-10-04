from dataclasses import asdict
from pathlib import Path

import torch

from .config import WaveFunctionConfig
from .wavefunction import WaveFunction

def save_checkpoint(
    path: str, 
    wf: WaveFunction, 
    optimizer: torch.optim, 
    wf_cfg: WaveFunctionConfig, 
    epochs: int, 
    metrics=None
):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try: 
        torch.save(
            {
                "model_state": wf.state_dict(),
                "optimizer_type": type(optimizer).__name__,
                "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
                "wf_cfg": asdict(wf_cfg),
                "epoch": epochs,
                "metrics": metrics or {},

            },
            path,
        )
    except Exception as e:
        print(f"❌ Error saving checkpoint: {e}")

def load_checkpoint(
    path: str,
    device="cpu"
):
    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}") 

    wf_cfg = WaveFunctionConfig(**checkpoint["wf_cfg"])
    
    #wf = model_cls(wf_cfg).to(device)
    #wf.load_state_dict(checkpoint["model_state"])

    #optimizer = None
    #if optimizer_cls is not None and checkpoint.get("optimizer_state") is not None:
    #    optimizer = optimizer_cls(wf.parameters(), lr=lr)
    #    optimizer.load_state_dict(checkpoint["optimizer_state"])

    return wf_cfg, checkpoint

def resolve_state_dict(model, checkpoint_state):
    model_state = model.state_dict()

    compatible = {}
    skipped = []

    for k, v in checkpoint_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append((k, tuple(v.shape), tuple(model_state[k].shape) if k in model_state else None))

    new_tensors = [
        k for k in model_state.keys()
        if k not in checkpoint_state
    ]

    model_state.update(compatible)
    model.load_state_dict(model_state)

    print("    State resolved:")
    print(f"    Loaded {len(compatible)} tensors")
    print(f"    Skipped {len(skipped)} tensors")
    for item in skipped:
        print("        skipped:", item)

    print(f"    Added {len(new_tensors)} new tensors")
    for item in new_tensors:
        print("        added:", item)
