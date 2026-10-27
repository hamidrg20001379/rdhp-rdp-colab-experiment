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
