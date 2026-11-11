# Accuracy comparison for the full-network benchmark

This report is derived from `all_seeds_method_results.csv`; raw results are unchanged.
The benchmark compares the pair-corrected RDP heuristic with additive RDP, direct RDP, and reconstructed RDHP.

## Post-fine-tuning accuracy

Values are mean ± sample SD across the available seeds. `Δ dense` is the method accuracy minus the dense accuracy.

| Target sparsity | Method | n | Immediate | 5 epochs | Δ dense | 20 epochs | Δ dense | Actual sparsity | FLOPs reduction |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50% | Additive RDP | 3 | 51.37 ± 3.78 | 54.83 ± 0.40 | +0.47 | 62.83 ± 3.23 | +8.47 | 49.98% | 35.31% |
| 50% | Pair-corrected RDP | 3 | 46.10 ± 9.33 | 56.13 ± 2.32 | +1.77 | 63.47 ± 3.44 | +9.10 | 49.94% | 36.07% |
| 50% | Direct RDP | 3 | 51.77 ± 3.44 | 54.53 ± 1.07 | +0.17 | 62.33 ± 2.40 | +7.97 | 49.96% | 35.13% |
| 50% | RDHP | 3 | 51.47 ± 2.53 | 57.63 ± 1.15 | +3.27 | 65.10 ± 1.78 | +10.73 | 49.87% | 29.41% |
| 75% | Additive RDP | 3 | 31.47 ± 6.82 | 56.27 ± 3.43 | +1.90 | 61.63 ± 1.57 | +7.27 | 75.01% | 60.90% |
| 75% | Pair-corrected RDP | 3 | 28.23 ± 12.17 | 54.83 ± 5.75 | +0.47 | 59.27 ± 1.48 | +4.90 | 75.00% | 61.93% |
| 75% | Direct RDP | 3 | 30.83 ± 7.78 | 56.43 ± 3.75 | +2.07 | 59.97 ± 2.58 | +5.60 | 75.03% | 62.61% |
| 75% | RDHP | 3 | 32.33 ± 4.94 | 55.40 ± 3.82 | +1.03 | 58.13 ± 3.50 | +3.77 | 74.97% | 47.93% |

## Paired differences involving the pair-corrected method

Positive accuracy values mean pair-corrected RDP is higher; negative distortion values mean pair-corrected RDP has lower test distortion.

| Target sparsity | Compared with | n | Immediate accuracy Δ | 5-epoch accuracy Δ | 20-epoch accuracy Δ | Test distortion Δ |
|---:|---|---:|---:|---:|---:|---:|
| 50% | Additive RDP | 3 | -5.27 | +1.30 | +0.63 | +0.6971 |
| 50% | Direct RDP | 3 | -5.67 | +1.60 | +1.13 | +0.7366 |
| 50% | RDHP | 3 | -5.37 | -1.50 | -1.63 | +0.6481 |
| 75% | Additive RDP | 3 | -3.23 | -1.43 | -2.37 | +0.5775 |
| 75% | Direct RDP | 3 | -2.60 | -1.60 | -0.70 | +0.6415 |
| 75% | RDHP | 3 | -4.10 | -0.57 | +1.13 | +0.7073 |

## Scope and interpretation

- Rows analyzed: 24; methods: Additive RDP, Pair-corrected RDP, Direct RDP, RDHP.
- Runs: 3 paired runs; seed identifiers are not present in the aggregate input CSV.
- This is a bounded ResNet-32/CIFAR-10 benchmark with 1,000 held-out test examples per seed and reconstructed RDP/RDHP implementations.
- The pair-corrected method is a heuristic that uses measured pair interactions at a reference pruning ratio; these results do not establish general superiority or a full paper reproduction.
