import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from train_cb import build_dataset, build_adapter_and_dfa, build_model, get_arg_parser
from logic_loss_tt import LogicLossModule


def split_indices(n, val_ratio=0.1, test_ratio=0.1, seed=0):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]
    return train_idx, val_idx, test_idx


def dataset_split(dataset, val_ratio=0.1, test_ratio=0.1, seed=0):
    train_idx, val_idx, test_idx = split_indices(len(dataset), val_ratio, test_ratio, seed)
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, val_idx), torch.utils.data.Subset(dataset, test_idx)


def eval_model(model, adapter, dfa, loader, device, mask_to_state_only=False):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_sat = 0.0
    total_violations = 0
    total_episodes = 0

    with torch.no_grad():
        for batch in loader:
            x, y, mask = [b.to(device) for b in batch]
            logits, sup_loss = model(x, targets=y, mask=mask)
            preds = logits.argmax(dim=-1)
            if isinstance(dfa, (list, tuple)):
                sats = [adapter.batch_check_dfa_sat(preds, d, mask_to_state_only=mask_to_state_only) for d in dfa]
                sat = torch.stack(sats, dim=0).min(dim=0).values
            else:
                sat = adapter.batch_check_dfa_sat(preds, dfa, mask_to_state_only=mask_to_state_only)
            violations = (sat < 0.5).sum().item()

            total_loss += sup_loss.item()
            total_batches += 1
            total_sat += sat.sum().item()
            total_violations += violations
            total_episodes += sat.numel()

    avg_loss = total_loss / max(1, total_batches)
    sat_rate = total_sat / max(1, total_episodes)
    violation_rate = total_violations / max(1, total_episodes)
    return {"supervised_loss": avg_loss, "satisfaction_rate": sat_rate, "violation_rate": violation_rate}


def train_model(args, device):
    dataset = build_dataset(args)
    train_ds, val_ds, test_ds = dataset_split(dataset, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    adapter, deep_dfa, raw_dfa = build_adapter_and_dfa(args, dataset)
    model = build_model(args, dataset, vocab_size=adapter.num_token_ids - 1)
    logic = LogicLossModule(
        deep_dfa=deep_dfa,
        adapter=adapter,
        mode='global',
        num_samples=args.num_samples,
        temperature=args.temperature,
        alpha=args.alpha,
        eps=getattr(args, "logic_eps", 1e-10),
        clamp_acceptance=not getattr(args, "no_logic_clamp", False),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = [b.to(device) for b in batch]
            loss = logic.compute_loss(model, batch, return_components=False)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % args.eval_every == 0:
            val_metrics = eval_model(model, adapter, raw_dfa, val_loader, device, mask_to_state_only=args.mask_to_state_only)
            print(f"epoch {epoch}: train_loss={total_loss/max(1,n_batches):.4f} val_loss={val_metrics['supervised_loss']:.4f} sat={val_metrics['satisfaction_rate']:.3f}")

    # final eval
    val_metrics = eval_model(model, adapter, raw_dfa, val_loader, device, mask_to_state_only=args.mask_to_state_only)
    test_metrics = eval_model(model, adapter, raw_dfa, test_loader, device, mask_to_state_only=args.mask_to_state_only)
    return model, adapter, raw_dfa, val_metrics, test_metrics


def parse_eval_args():
    parent = get_arg_parser(add_help=False)
    parser = argparse.ArgumentParser(parents=[parent], description="Train/evaluate with splits")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="eval_runs")
    parser.add_argument("--mask_to_state_only", action="store_true", help="Only consider state symbols for DFA satisfaction")
    return parser


def main():
    parser = parse_eval_args()
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, adapter, dfa, val_metrics, test_metrics = train_model(args, device)

    os.makedirs(args.out_dir, exist_ok=True)
    result = {"val": val_metrics, "test": test_metrics}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    # CSV
    import csv
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["split", "supervised_loss", "satisfaction_rate", "violation_rate"])
        for split in ["val", "test"]:
            m = result[split]
            writer.writerow([split, m["supervised_loss"], m["satisfaction_rate"], m["violation_rate"]])

    # Simple plot of satisfaction rate
    try:
        import matplotlib.pyplot as plt
        splits = ["val", "test"]
        sats = [result[s]["satisfaction_rate"] for s in splits]
        plt.bar(splits, sats)
        plt.ylabel("Satisfaction rate")
        plt.savefig(os.path.join(args.out_dir, "satisfaction.png"))
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

    print("Saved metrics to", args.out_dir)


if __name__ == "__main__":
    main()
