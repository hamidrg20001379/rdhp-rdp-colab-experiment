#!/usr/bin/env python3
"""Summarize post-pruning accuracy and paired comparisons with pair-corrected RDP.

This script analyzes an already completed full-network benchmark. It does not
rerun training or alter the raw result CSV. Positive accuracy deltas in the
pairwise comparison mean that pair-corrected RDP is more accurate for that
seed/target than the comparison method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METHOD_ORDER = ["additive_rdp", "pair_corrected_rdp", "direct_rdp", "rdhp"]
METHOD_LABELS = {
    "additive_rdp": "Additive RDP",
    "pair_corrected_rdp": "Pair-corrected RDP",
    "direct_rdp": "Direct RDP",
    "rdhp": "RDHP",
}

REQUIRED_COLUMNS = {
    "target_sparsity",
    "method",
    "test_mean_distortion",
    "immediate_accuracy",
    "dense_accuracy",
    "finetuned_accuracy_5",
    "finetuned_accuracy_20",
    "sparsity_actual",
    "flops_reduction_fraction",
}


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["target_sparsity", "method"], as_index=False)
        .agg(
            n=("method", "size"),
            dense_accuracy_mean=("dense_accuracy", "mean"),
            dense_accuracy_sd=("dense_accuracy", "std"),
            immediate_accuracy_mean=("immediate_accuracy", "mean"),
            immediate_accuracy_sd=("immediate_accuracy", "std"),
            finetuned_accuracy_5_mean=("finetuned_accuracy_5", "mean"),
            finetuned_accuracy_5_sd=("finetuned_accuracy_5", "std"),
            finetuned_accuracy_20_mean=("finetuned_accuracy_20", "mean"),
            finetuned_accuracy_20_sd=("finetuned_accuracy_20", "std"),
            test_distortion_mean=("test_mean_distortion", "mean"),
            test_distortion_sd=("test_mean_distortion", "std"),
            actual_sparsity_mean=("sparsity_actual", "mean"),
            actual_sparsity_sd=("sparsity_actual", "std"),
            flops_reduction_mean=("flops_reduction_fraction", "mean"),
            flops_reduction_sd=("flops_reduction_fraction", "std"),
        )
    )
    grouped["method_order"] = grouped["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    return grouped.sort_values(["target_sparsity", "method_order"]).drop(columns="method_order")


def pairwise_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    pair = frame[frame["method"] == "pair_corrected_rdp"].copy()
    comparisons = []
    for method in METHOD_ORDER:
        if method == "pair_corrected_rdp":
            continue
        other = frame[frame["method"] == method].copy()
        keys = ["target_sparsity", "run_index"]
        merged = pair.merge(other, on=keys, suffixes=("_pair", "_other"), validate="one_to_one")
        for _, row in merged.iterrows():
            comparisons.append(
                {
                    "target_sparsity": row["target_sparsity"],
                    "run_index": row["run_index"],
                    "comparison_method": method,
                    "pair_minus_other_immediate_accuracy": row["immediate_accuracy_pair"] - row["immediate_accuracy_other"],
                    "pair_minus_other_finetuned_accuracy_5": row["finetuned_accuracy_5_pair"] - row["finetuned_accuracy_5_other"],
                    "pair_minus_other_finetuned_accuracy_20": row["finetuned_accuracy_20_pair"] - row["finetuned_accuracy_20_other"],
                    "pair_minus_other_test_distortion": row["test_mean_distortion_pair"] - row["test_mean_distortion_other"],
                    "pair_minus_other_flops_reduction": row["flops_reduction_fraction_pair"] - row["flops_reduction_fraction_other"],
                }
            )
    return pd.DataFrame(comparisons)


def write_report(frame: pd.DataFrame, summary: pd.DataFrame, deltas: pd.DataFrame, output: Path) -> None:
    lines = [
        "# Accuracy comparison for the full-network benchmark",
        "",
        "This report is derived from `all_seeds_method_results.csv`; raw results are unchanged.",
        "The benchmark compares the pair-corrected RDP heuristic with additive RDP, direct RDP, and reconstructed RDHP.",
        "",
        "## Post-fine-tuning accuracy",
        "",
        "Values are mean ± sample SD across the available seeds. `Δ dense` is the method accuracy minus the dense accuracy.",
        "",
        "| Target sparsity | Method | n | Immediate | 5 epochs | Δ dense | 20 epochs | Δ dense | Actual sparsity | FLOPs reduction |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        ft5_delta = row.finetuned_accuracy_5_mean - row.dense_accuracy_mean
        ft20_delta = row.finetuned_accuracy_20_mean - row.dense_accuracy_mean
        lines.append(
            f"| {row.target_sparsity:.0%} | {METHOD_LABELS[row.method]} | {row.n} | "
            f"{fmt(row.immediate_accuracy_mean)} ± {fmt(row.immediate_accuracy_sd)} | "
            f"{fmt(row.finetuned_accuracy_5_mean)} ± {fmt(row.finetuned_accuracy_5_sd)} | "
            f"{ft5_delta:+.2f} | {fmt(row.finetuned_accuracy_20_mean)} ± {fmt(row.finetuned_accuracy_20_sd)} | "
            f"{ft20_delta:+.2f} | {row.actual_sparsity_mean:.2%} | {row.flops_reduction_mean:.2%} |"
        )

    lines += [
        "",
        "## Paired differences involving the pair-corrected method",
        "",
        "Positive accuracy values mean pair-corrected RDP is higher; negative distortion values mean pair-corrected RDP has lower test distortion.",
        "",
        "| Target sparsity | Compared with | n | Immediate accuracy Δ | 5-epoch accuracy Δ | 20-epoch accuracy Δ | Test distortion Δ |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    delta_summary = (
        deltas.groupby(["target_sparsity", "comparison_method"], as_index=False)
        .agg(
            n=("comparison_method", "size"),
            immediate=("pair_minus_other_immediate_accuracy", "mean"),
            ft5=("pair_minus_other_finetuned_accuracy_5", "mean"),
            ft20=("pair_minus_other_finetuned_accuracy_20", "mean"),
            distortion=("pair_minus_other_test_distortion", "mean"),
        )
    )
    delta_summary["method_order"] = delta_summary["comparison_method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    for row in delta_summary.sort_values(["target_sparsity", "method_order"]).itertuples(index=False):
        lines.append(
            f"| {row.target_sparsity:.0%} | {METHOD_LABELS[row.comparison_method]} | {row.n} | "
            f"{row.immediate:+.2f} | {row.ft5:+.2f} | {row.ft20:+.2f} | {row.distortion:+.4f} |"
        )

    lines += [
        "",
        "## Scope and interpretation",
        "",
        f"- Rows analyzed: {len(frame)}; methods: {', '.join(METHOD_LABELS[m] for m in METHOD_ORDER)}.",
        f"- Runs: {frame['run_index'].nunique()} paired runs; seed identifiers are not present in the aggregate input CSV.",
        "- This is a bounded ResNet-32/CIFAR-10 benchmark with 1,000 held-out test examples per seed and reconstructed RDP/RDHP implementations.",
        "- The pair-corrected method is a heuristic that uses measured pair interactions at a reference pruning ratio; these results do not establish general superiority or a full paper reproduction.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    unknown = set(frame["method"].unique()) - set(METHOD_ORDER)
    if unknown:
        raise ValueError(f"Unexpected methods in input: {sorted(unknown)}")
    if "seed" in frame.columns:
        frame["run_index"] = frame["seed"]
    else:
        # The published aggregate CSV omits seed IDs but preserves complete
        # four-method blocks for each run and target sparsity.
        method_count = frame["method"].nunique()
        frame["run_index"] = frame.groupby("target_sparsity").cumcount() // method_count

    summary = summary_table(frame)
    deltas = pairwise_deltas(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "accuracy_summary.csv", index=False)
    deltas.to_csv(args.output_dir / "pairwise_accuracy_deltas.csv", index=False)
    write_report(frame, summary, deltas, args.output_dir / "accuracy_comparison_report.md")
    print(f"Wrote {args.output_dir / 'accuracy_summary.csv'}")
    print(f"Wrote {args.output_dir / 'pairwise_accuracy_deltas.csv'}")
    print(f"Wrote {args.output_dir / 'accuracy_comparison_report.md'}")


if __name__ == "__main__":
    main()
