#!/usr/bin/env python3
"""Run the pairwise-corrected full-network benchmark on CIFAR models.

The allocation and evaluation protocol matches the ResNet-32 benchmark:
additive RDP, pair-corrected RDP, direct RDP, and reconstructed RDHP are
evaluated at matched sparsity targets, with immediate and fine-tuned accuracy.
Each model/seed writes isolated raw artifacts and the aggregate CSV is written
only after every requested run completes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cifar_model_zoo import MODEL_NAMES, load_model, model_description


def import_algorithm_modules(thesis_root: Path):
    experiments = thesis_root / "working" / "experiments"
    if str(experiments) not in sys.path:
        sys.path.insert(0, str(experiments))
    from rdp_resnet32_adapter import (
        accuracy,
        allocate_dp,
        apply_masks,
        build_curves as build_rdp_curves,
        load_checkpoint,
        logits,
        make_loaders,
        model_stats,
        prunable_layers,
        seed_everything,
        train_epochs,
    )
    from rdhp_resnet32_adapter import build_rdhp_curves
    return Namespace(
        accuracy=accuracy,
        allocate_dp=allocate_dp,
        apply_masks=apply_masks,
        build_rdp_curves=build_rdp_curves,
        build_rdhp_curves=build_rdhp_curves,
        load_checkpoint=load_checkpoint,
        logits=logits,
        make_loaders=make_loaders,
        model_stats=model_stats,
        prunable_layers=prunable_layers,
        seed_everything=seed_everything,
        train_epochs=train_epochs,
    )


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def mask_topk(weight: torch.Tensor, fraction: float) -> torch.Tensor:
    flat = weight.detach().abs().flatten()
    keep = max(1, min(flat.numel(), int(round((1.0 - float(fraction)) * flat.numel()))))
    order = torch.argsort(flat, descending=True, stable=True)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[order[:keep]] = True
    return mask.reshape_as(weight)


def all_one_masks(model: nn.Module, ops) -> list[torch.Tensor]:
    return [torch.ones_like(layer.weight, dtype=torch.bool).cpu() for layer in ops.prunable_layers(model)]


def measure_mask(model, masks, calibration_loader, test_loader, dense_calib, dense_test, device, ops):
    masked = copy.deepcopy(model).to(device)
    ops.apply_masks(masked, masks)
    masked_calib = ops.logits(masked, calibration_loader, device)
    masked_test = ops.logits(masked, test_loader, device)
    calib_error = (masked_calib - dense_calib).flatten(1).square().mean(dim=1)
    test_error = (masked_test - dense_test).flatten(1).square().mean(dim=1)
    result = {
        "calib_mean_distortion": float(calib_error.mean()),
        "test_mean_distortion": float(test_error.mean()),
        "calib_max_distortion": float(calib_error.max()),
        "test_max_distortion": float(test_error.max()),
    }
    del masked, masked_calib, masked_test
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def measure_calib_mean(model, masks, calibration_loader, dense_calib, device, ops):
    masked = copy.deepcopy(model).to(device)
    ops.apply_masks(masked, masks)
    masked_calib = ops.logits(masked, calibration_loader, device)
    error = (masked_calib - dense_calib).flatten(1).square().mean(dim=1)
    value = float(error.mean())
    del masked, masked_calib
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return value


def estimate_flops(model, masks, loader, device, ops):
    records = []
    handles = []

    def hook(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            output_elements = int(output.numel())
            kernel_ops = (module.in_channels // module.groups) * int(np.prod(module.kernel_size))
            dense_ops = output_elements * kernel_ops
        elif isinstance(module, nn.Linear):
            dense_ops = int(output.numel()) * module.in_features
        else:
            dense_ops = 0
        active_fraction = float(masks[len(records)].float().mean())
        records.append((dense_ops, dense_ops * active_fraction))

    for layer in ops.prunable_layers(model):
        handles.append(layer.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    try:
        x, _ = next(iter(loader))
        with torch.no_grad():
            model(x.to(device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    dense_flops = sum(item[0] for item in records)
    sparse_flops = sum(item[1] for item in records)
    return {
        "dense_flops_per_batch": float(dense_flops),
        "sparse_flops_per_batch": float(sparse_flops),
        "flops_retained_fraction": float(sparse_flops / max(dense_flops, 1.0)),
        "flops_reduction_fraction": float(1.0 - sparse_flops / max(dense_flops, 1.0)),
    }


def prepare_allocations(model, calibration_loader, dense_calib, device, seed, config, ops, layer_names, pair_indices):
    started = time.monotonic()
    ratios, worst_curves = ops.build_rdp_curves(
        model, calibration_loader, device,
        curve_points=config.curve_points, max_pruning=1.0,
        worst_case=True, monotonic_repair=True,
    )
    mean_ratios, mean_curves = ops.build_rdp_curves(
        model, calibration_loader, device,
        curve_points=config.curve_points, max_pruning=1.0,
        worst_case=False, monotonic_repair=False,
    )
    if not np.allclose(ratios, mean_ratios):
        raise RuntimeError("RDP curve grids do not match")

    reference_index = int(np.argmin(np.abs(ratios - config.pair_reference_ratio)))
    pair_rows = []
    corrected_curves = mean_curves.copy()
    layers = ops.prunable_layers(model)
    for layer_a, layer_b in pair_indices:
        masks_ab = all_one_masks(model, ops)
        masks_ab[layer_a] = mask_topk(layers[layer_a].weight, config.pair_reference_ratio).detach().cpu()
        masks_ab[layer_b] = mask_topk(layers[layer_b].weight, config.pair_reference_ratio).detach().cpu()
        single_a = float(mean_curves[layer_a, reference_index])
        single_b = float(mean_curves[layer_b, reference_index])
        joint = measure_calib_mean(model, masks_ab, calibration_loader, dense_calib, device, ops)
        interaction = joint - single_a - single_b
        pair_rows.append({
            "layer_a": layer_a, "layer_a_name": layer_names[layer_a],
            "layer_b": layer_b, "layer_b_name": layer_names[layer_b],
            "reference_ratio": config.pair_reference_ratio,
            "single_a": single_a, "single_b": single_b,
            "joint": joint, "interaction": interaction,
        })
        for layer_index in (layer_a, layer_b):
            contribution = interaction / 2.0
            scale = mean_curves[layer_index] / max(float(mean_curves[layer_index, reference_index]), 1e-12)
            corrected_curves[layer_index] += contribution * scale
    corrected_curves = np.maximum(corrected_curves, 0.0)

    rdhp_ratios, rdhp_curves, fishers = ops.build_rdhp_curves(
        model, calibration_loader, device,
        curve_points=config.curve_points, max_pruning=1.0,
        fisher_probes=config.fisher_probes, fisher_damping=config.fisher_damping,
        fisher_mode="diagonal", dense_limit=2048, seed=seed,
    )
    if not np.allclose(ratios, rdhp_ratios):
        raise RuntimeError("RDP and RDHP curve grids do not match")

    allocations = {}
    for target in config.target_sparsities:
        allocations[float(target)] = {}
        for method, curves in (
            ("additive_rdp", worst_curves),
            ("direct_rdp", mean_curves),
            ("pair_corrected_rdp", corrected_curves),
            ("rdhp", rdhp_curves),
        ):
            allocations[float(target)][method] = ops.allocate_dp(
                model, ratios, curves, target, budget_bins=config.budget_bins
            )
    return {
        "ratios": ratios,
        "worst_curves": worst_curves,
        "mean_curves": mean_curves,
        "corrected_curves": corrected_curves,
        "rdhp_curves": rdhp_curves,
        "fishers": fishers,
        "pair_rows": pair_rows,
        "allocations": allocations,
        "elapsed_seconds": time.monotonic() - started,
    }


def resolve_checkpoint(pattern: str | None, model_name: str, seed: int) -> Path | None:
    if not pattern:
        return None
    return Path(pattern.format(model=model_name, seed=seed))


def run_seed(model_name, seed, config, ops, output_dir, pair_indices, layer_names):
    ops.seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_args = Namespace(
        synthetic=config.synthetic, data_root=config.data_root, download=config.download,
        train_samples=config.train_samples, calib_samples=config.calib_samples,
        test_samples=config.test_samples, batch_size=config.batch_size, seed=seed,
    )
    (train_loader, calibration_loader, test_loader), split_info = ops.make_loaders(loader_args)
    model = load_model(model_name, thesis_root=config.thesis_root).to(device)
    checkpoint = resolve_checkpoint(config.checkpoint_pattern, model_name, seed)
    checkpoint_sha256 = None
    dense_source = "random_initialization"
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Dense checkpoint not found: {checkpoint}")
        checkpoint_sha256 = ops.load_checkpoint(model, checkpoint)
        dense_source = str(checkpoint)
    elif config.train_epochs > 0:
        ops.train_epochs(model, train_loader, device, config.train_epochs, config.learning_rate, seed)
        dense_source = f"trained_in_runner:{config.train_epochs}_epochs"
    else:
        raise ValueError("Provide --checkpoint-pattern or set --train-epochs > 0")

    dense_calib = ops.logits(model, calibration_loader, device)
    dense_test = ops.logits(model, test_loader, device)
    dense_accuracy = ops.accuracy(model, test_loader, device)
    prepared = prepare_allocations(model, calibration_loader, dense_calib, device, seed, config, ops, layer_names, pair_indices)
    rows = []
    for target, methods in prepared["allocations"].items():
        measured = {}
        for method, (masks, allocation_info) in methods.items():
            values = measure_mask(model, masks, calibration_loader, test_loader, dense_calib, dense_test, device, ops)
            masked = copy.deepcopy(model).to(device)
            ops.apply_masks(masked, masks)
            immediate = ops.accuracy(masked, test_loader, device)
            stats = ops.model_stats(masked)
            flops = estimate_flops(masked, masks, test_loader, device, ops)
            measured[method] = {
                "model": model_name, "seed": seed, "target_sparsity": target, "method": method,
                **values, "immediate_accuracy": immediate, "dense_accuracy": dense_accuracy,
                "accuracy_delta_immediate": immediate - dense_accuracy, **stats, **flops,
                "allocation_achieved_sparsity": allocation_info["achieved_sparsity"],
            }
            for epochs in config.finetune_epochs:
                finetuned = copy.deepcopy(masked).to(device)
                ops.train_epochs(finetuned, train_loader, device, epochs, config.learning_rate, seed + 1000 + epochs, masks=masks)
                ops.apply_masks(finetuned, masks)
                final_accuracy = ops.accuracy(finetuned, test_loader, device)
                measured[method][f"finetuned_accuracy_{epochs}"] = final_accuracy
                measured[method][f"accuracy_delta_finetuned_{epochs}"] = final_accuracy - dense_accuracy
                del finetuned
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            rows.append(measured[method])
            del masked
            if device.type == "cuda":
                torch.cuda.empty_cache()
        oracle = min(row["test_mean_distortion"] for row in measured.values())
        for row in rows[-len(measured):]:
            row["test_distortion_regret"] = (row["test_mean_distortion"] - oracle) / max(oracle, 1e-12)
            row["test_distortion_oracle"] = oracle

    output_dir.mkdir(parents=True, exist_ok=False)
    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "method_results.csv", index=False)
    pd.DataFrame(prepared["pair_rows"]).to_csv(output_dir / "pair_interactions.csv", index=False)
    np.savez_compressed(
        output_dir / "allocation_curves.npz",
        ratios=prepared["ratios"], worst_curves=prepared["worst_curves"],
        mean_curves=prepared["mean_curves"], corrected_curves=prepared["corrected_curves"],
        rdhp_curves=prepared["rdhp_curves"],
    )
    torch.save({"fishers": prepared["fishers"]}, output_dir / "rdhp_fishers.pt")
    accuracy_column = f"finetuned_accuracy_{max(config.finetune_epochs)}"
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for target in config.target_sparsities:
        plot_frame = results_df[results_df["target_sparsity"] == target]
        axes[0].plot(plot_frame["method"], plot_frame["test_mean_distortion"], marker="o", label=f"{target:.0%} sparsity")
        axes[1].plot(plot_frame["method"], plot_frame[accuracy_column], marker="o", label=f"{target:.0%} sparsity")
    axes[0].set_ylabel("Test mean logit distortion")
    axes[0].set_title(f"{model_name}, seed {seed}: distortion")
    axes[0].legend()
    axes[1].set_ylabel("Test accuracy (%)")
    axes[1].set_title(f"{model_name}, seed {seed}: accuracy after fine-tuning")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "full_network_comparison.png", dpi=160)
    plt.close(figure)
    (output_dir / "metadata.json").write_text(json.dumps({
        "model": model_name, "model_description": model_description(model_name),
        "seed": seed, "dense_source": dense_source, "dense_checkpoint_sha256": checkpoint_sha256,
        "split": split_info, "layer_names": layer_names, "pair_indices": pair_indices,
        "curve_build_seconds": prepared["elapsed_seconds"], "config": vars(config),
    }, indent=2, default=jsonable) + "\n", encoding="utf-8")
    print(f"{model_name} seed {seed} complete: dense={dense_accuracy:.2f}%", flush=True)
    return results_df


def self_test(thesis_root: Path | None) -> None:
    for name in MODEL_NAMES:
        model = load_model(name, thesis_root=thesis_root)
        output = model(torch.randn(2, 3, 32, 32))
        assert tuple(output.shape) == (2, 10)
        layers = [layer for layer in model.modules() if isinstance(layer, (nn.Conv2d, nn.Linear))]
        assert layers
        print(f"PASS {name}: output={tuple(output.shape)}, prunable_layers={len(layers)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, nargs="+", default=list(MODEL_NAMES))
    parser.add_argument("--thesis-root", type=Path, default=Path("/workspace/my-thesis"))
    parser.add_argument("--data-root", default="/workspace/my-thesis/data.cifar10")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--calib-samples", type=int, default=1024)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-epochs", type=int, default=0)
    parser.add_argument("--checkpoint-pattern", help="Path template with {model} and {seed} placeholders")
    parser.add_argument("--target-sparsities", type=float, nargs="+", default=[0.50, 0.75])
    parser.add_argument("--curve-points", type=int, default=20)
    parser.add_argument("--budget-bins", type=int, default=5000)
    parser.add_argument("--pair-count", type=int, default=24)
    parser.add_argument("--pair-reference-ratio", type=float, default=0.50)
    parser.add_argument("--fisher-probes", type=int, default=2)
    parser.add_argument("--fisher-damping", type=float, default=1e-7)
    parser.add_argument("--finetune-epochs", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(args.thesis_root)
        return
    if args.output is None:
        raise ValueError("real runs require --output")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if not args.synthetic and args.checkpoint_pattern is None and args.train_epochs <= 0:
        raise ValueError("Real runs need --checkpoint-pattern or --train-epochs > 0")
    args.output.mkdir(parents=True)
    ops = import_algorithm_modules(args.thesis_root)
    all_results = []
    for model_name in args.model:
        probe = load_model(model_name, thesis_root=args.thesis_root)
        layer_names = [name for name, module in probe.named_modules() if isinstance(module, (nn.Conv2d, nn.Linear))]
        layer_count = len(layer_names)
        rng = np.random.default_rng(123)
        all_pairs = [(a, b) for a in range(layer_count) for b in range(a + 1, layer_count)]
        selected = rng.choice(len(all_pairs), size=min(args.pair_count, len(all_pairs)), replace=False)
        pair_indices = [all_pairs[int(index)] for index in selected]
        del probe
        model_output = args.output / model_name
        model_output.mkdir()
        for seed in args.seeds:
            all_results.append(run_seed(model_name, seed, args, ops, model_output / f"seed_{seed}", pair_indices, layer_names))
    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(args.output / "all_models_method_results.csv", index=False)
    numeric = [column for column in results.columns if column.startswith(("immediate_accuracy", "finetuned_accuracy_", "test_", "sparsity", "flops_reduction_fraction", "accuracy_delta"))]
    results.groupby(["model", "target_sparsity", "method"])[numeric].agg(["mean", "std"]).to_csv(args.output / "all_models_mean_std.csv")
    (args.output / "summary.json").write_text(json.dumps({
        "models": args.model, "seeds": args.seeds,
        "target_sparsities": args.target_sparsities,
        "model_descriptions": {name: model_description(name) for name in args.model},
        "results": results.to_dict(orient="records"),
        "protocol": vars(args),
    }, indent=2, default=jsonable) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
