from mbp.checkpoint import load_checkpoint
from mbp.utils import print_section

import argparse

def show():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "filepath",
        type=str,
        help="Path to checkpoint file (.pt or .pth)"
    )

    parser.add_argument(
        "-m", "--model",
        action="store_true",
        help="Display model layers and parameter counts"
    )

    parser.add_argument(
        "-p", "--parameters",
        action="store_true",
        help="Display all model parameters"
    )

    args = parser.parse_args()
    FILE = args.filepath

    wf_cfg, checkpoint = load_checkpoint(FILE, device="cpu")
    #checkpoint = load_checkpoint(FILE, device="cpu")
    
    print_section("Checkpoint Details", f"{FILE}")

    print("   ┌Wave Function Configuration")
    for attr, value in wf_cfg.__dict__.items():
        print(f"   │ {attr}: {value}")

    if args.model:
        from torch import nn
        from mbp.config import WaveFunctionConfig
        from mbp.wavefunction import WaveFunction

        wf = WaveFunction(wf_cfg)
        wf.load_state_dict(checkpoint["model_state"])
        layers = nn.ModuleList(wf.children())
        print("   ┌Neural Network Information")
        for layer in layers:
            params = sum(p.numel() for p in layer.parameters())
            print(f"   │ {layer}: {params} parameters")

            if args.parameters:
                for p in layer.parameters():
                    print(f"   │ {p}") 


    print("   ┌Energy Information")
    for attr, value in checkpoint["metrics"].items():
        print(f"   │ {attr}: {value}")

    print("   ┌Optimization Information")
    print(f'   │ Optimizer: {"sr" if None else checkpoint["optimizer_type"]}')
    print(f'   │ epochs: {checkpoint["epoch"]}')


if __name__ == "__main__":
    show()
