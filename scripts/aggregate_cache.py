#!/usr/bin/env python3
"""Aggregate per-replicate .pt cache files into summary text files.

After SLURM array jobs finish, run this to collect results:

    python scripts/aggregate_cache.py --chapter 7
    python scripts/aggregate_cache.py --chapter 8
    python scripts/aggregate_cache.py --chapter 9
    python scripts/aggregate_cache.py --chapter 14
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch


def _summarize(values: list[float]) -> str:
    """Format mean ± SE."""
    arr = np.array(values)
    mean = arr.mean()
    se = arr.std() / np.sqrt(len(arr))
    return f"{mean:.5f} +/- {se:.5f}"


def aggregate_ch07():
    """Aggregate Ch 7 surrogates results."""
    for dataset in ["bgp", "friedman"]:
        files = sorted(glob.glob(f"cache/ch07_{dataset}_rep*.pt"))
        if not files:
            print(f"  No files found for ch07_{dataset}")
            continue

        all_results = [torch.load(f, weights_only=False) for f in files]
        print(f"  {dataset}: {len(files)} replicates")

        outpath = f"cache/{dataset}_numbers.txt"
        lines = []

        if dataset == "bgp":
            for d_key in ["d2", "d3", "d4"]:
                vals = [r[d_key] for r in all_results if d_key in r]
                if not vals:
                    continue
                for metric in ["gbc_rmse", "gbc_crps", "gbc_cov", "gp_rmse", "gp_crps", "gp_cov"]:
                    arr = [v[metric] for v in vals]
                    lines.append(f"{d_key}_{metric} = {_summarize(arr)}")
        else:
            for metric in ["gbc_rmse", "gbc_crps", "gbc_cov", "gp_rmse", "gp_crps"]:
                arr = [r[metric] for r in all_results]
                lines.append(f"{metric} = {_summarize(arr)}")

        with open(outpath, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  -> {outpath}")


def aggregate_ch08():
    """Aggregate Ch 8 jump results."""
    for dataset in ["phantom", "star"]:
        files = sorted(glob.glob(f"cache/ch08_{dataset}_rep*.pt"))
        if not files:
            print(f"  No files found for ch08_{dataset}")
            continue

        all_results = [torch.load(f, weights_only=False) for f in files]
        print(f"  {dataset}: {len(files)} replicates")

        outpath = f"cache/augiqn_{dataset}_numbers.txt"
        lines = []
        for metric in ["rmse", "crps", "coverage"]:
            arr = [r[metric] for r in all_results]
            lines.append(f"{metric} = {_summarize(arr)}")

        with open(outpath, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  -> {outpath}")


def aggregate_ch09():
    """Aggregate Ch 9 active learning results."""
    files = sorted(glob.glob("cache/ch09_rocket_rep*.pt"))
    if not files:
        print("  No files found for ch09_rocket")
        return

    all_results = [torch.load(f, weights_only=False) for f in files]
    print(f"  rocket: {len(files)} replicates")

    outpath = "cache/al_numbers.txt"
    lines = []
    for metric in ["rmse", "crps", "coverage"]:
        arr = [r[metric] for r in all_results]
        lines.append(f"{metric} = {_summarize(arr)}")

    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {outpath}")


def aggregate_ch14():
    """Ch 14 produces cache/lake_numbers.txt directly — verify it exists."""
    if os.path.exists("cache/lake_numbers.txt"):
        with open("cache/lake_numbers.txt") as f:
            print(f"  lake_numbers.txt exists:\n{f.read()}")
    else:
        print("  cache/lake_numbers.txt not found.")
        print("  Ch 14 writes this file directly; run scripts/ch14_lake.py first.")


CHAPTER_MAP = {
    7: ("Chapter 7: Surrogates", aggregate_ch07),
    8: ("Chapter 8: Jumps", aggregate_ch08),
    9: ("Chapter 9: Active Learning", aggregate_ch09),
    14: ("Chapter 14: Lake Calibration", aggregate_ch14),
}


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-replicate .pt files into summary text files")
    parser.add_argument("--chapter", type=int, required=True, choices=[7, 8, 9, 14],
                        help="Chapter number to aggregate")
    args = parser.parse_args()

    title, fn = CHAPTER_MAP[args.chapter]
    print(f"\n{title}")
    print("-" * len(title))
    fn()
    print()


if __name__ == "__main__":
    main()
