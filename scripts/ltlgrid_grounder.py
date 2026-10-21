"""Train and evaluate the symbol grounder g(state, action[, rtg]) -> symbol on an ltlgrid H5 dataset."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from datasets.ltlgrid_dataset import LTLGridSequenceDataset
from planning.ltlgrid_logic import train_grounder


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True, help="dataset_*.h5 (the task is read from a 'task<N>' path component)")
    p.add_argument("--task_id", type=int, default=None, help="0-based task, overrides the path")
    p.add_argument("--mode", choices=["oracle", "supervised", "weak"], default="supervised")
    p.add_argument("--use_rtg", action="store_true", help="also feed the return-to-go to the grounder")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="write the stats as JSON here")
    args = p.parse_args(argv)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = LTLGridSequenceDataset(args.h5, task_id=args.task_id)
    grounder, stats = train_grounder(
        dataset,
        mode=args.mode,
        use_rtg=args.use_rtg,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        seed=args.seed,
        device=device,
    )
    stats["task_id"] = dataset.task_id
    print(json.dumps(stats, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    return grounder, stats


if __name__ == "__main__":
    main()
