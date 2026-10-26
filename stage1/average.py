"""Uniform weight average of finetuning epochs1-5, including floating BN buffers."""
import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs=5, type=Path)
    parser.add_argument("--out", type=Path, default=Path("work/final.pt"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    average = None
    for index, path in enumerate(args.checkpoints):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["epoch"] != index + 1:
            raise ValueError("Supply finetuning epochs 1, 2, 3, 4, 5 in order")
        state = checkpoint["model"]
        if average is None:
            # Accumulate parameters and floating buffers in FP64. Integer
            # counters cannot be averaged; keep their last-epoch values.
            average = {k: torch.zeros_like(v, dtype=torch.float64) if v.is_floating_point() else v.clone()
                       for k, v in state.items()}
        if set(state) != set(average):
            raise ValueError("Model tensors differ")
        for key, value in state.items():
            if value.is_floating_point():
                average[key].add_(value.double())
            else:
                average[key] = value.clone()
    average = {k: (v / 5).to(state[k].dtype) if v.is_floating_point() else v for k, v in average.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": average, "cfg": checkpoint["cfg"], "averaged_epochs": [1, 2, 3, 4, 5]}, args.out)


if __name__ == "__main__":
    main()
