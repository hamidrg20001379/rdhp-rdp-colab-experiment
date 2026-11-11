# RDHP/RDP Colab experiment

Import `rdhp_rdp_colab_experiment.ipynb` into Google Colab to run the
reconstructed TPAMI 2025 RDHP/RDP ResNet-32 CIFAR-10 experiments.

The notebook clones `hamidrg20001379/my-thesis` with its submodules, runs the
software checks and synthetic smoke test, and provides an opt-in real-data
pilot using a shared dense checkpoint. Add a Colab Secret named
`GITHUB_TOKEN` with private-repository read access before cloning.

The real-data run is disabled by default. The notebook records achieved
sparsity and method metadata, but it does not turn a pilot into a claimed
paper reproduction.

## Cross-model pairwise benchmark

`run_cross_model_pairwise.py` runs the same full-network comparison for
`resnet32`, `vgg16`, and `densenet121`. Each model is evaluated with additive
RDP, pair-corrected RDP, direct RDP, and reconstructed RDHP at the requested
sparsity targets. It records pair interactions, allocation curves, achieved
sparsity, approximate FLOPs, immediate accuracy, and fine-tuned accuracy.

The ResNet-32 model is loaded from the pinned source in the thesis checkout.
VGG-16 and DenseNet-121 use the CIFAR-compatible definitions in
`cifar_model_zoo.py`; these are explicit adaptations and should not be called
exact author-code reproductions.

Run the structural smoke test with:

```sh
python run_cross_model_pairwise.py --self-test
```

For real data, prepare a dense checkpoint for every requested model and seed,
then use a path template containing `{model}` and `{seed}`:

```sh
python run_cross_model_pairwise.py \
  --model resnet32 vgg16 densenet121 \
  --thesis-root /content/my-thesis \
  --checkpoint-pattern '/content/drive/MyDrive/cross_model_checkpoints/{model}/seed_{seed}/dense.pt' \
  --data-root /content/drive/MyDrive/cifar10 \
  --seeds 42 43 44 \
  --output /content/drive/MyDrive/cross_model_pairwise_all_seeds
```

If checkpoints are not available, set `--train-epochs 20` to train each dense
model inside the runner. Use a fresh output directory for every run. The
synthetic smoke artifacts under `results/` are pipeline validation only.

The ready-to-import notebook `vgg16_densenet121_pairwise_all_seeds.ipynb` runs
only VGG-16 and DenseNet-121; it intentionally does not rerun ResNet-32.
