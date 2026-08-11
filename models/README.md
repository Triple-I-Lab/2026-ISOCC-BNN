# Binary Neural Networks on PyTorch

![Binarization](https://github.com/lucamocerino/Binary-Neural-Networks-PyTorch-1.0/blob/master/bin.png)

This repository implements three popular approaches to Binary Neural Networks (BNNs) in PyTorch:

- [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279)
- [Binarized Neural Networks](https://papers.nips.cc/paper/6573-binarized-neural-networks)
- [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160)

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Models](#models)
- [Binary Layer Implementations](#binary-layer-implementations)
- [Classifiers](#classifiers)
- [Datasets](#datasets)
- [Configuration Files](#configuration-files)
- [Running the Tests](#running-the-tests)
- [Related Applications](#related-applications)
- [License](#license)

---

## Installation

```sh
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`):

| Package | Purpose |
|---|---|
| `torch` | Core deep-learning framework |
| `torchvision` | Dataset loaders (MNIST, CIFAR-10) |
| `tqdm` | Progress bars during training |
| `pyyaml` | YAML configuration file parsing |

---

## Quick Start

Training is driven by a YAML configuration file passed on the command line:

```sh
python main.py app:<yml_file>
```

**Examples**

| Command | Model | Dataset |
|---|---|---|
| `python main.py app:yml/nin_cifar10.yml` | XNOR Network-in-Network | CIFAR-10 |
| `python main.py app:yml/bnn_caffenet_cifar10.yml` | BNN CaffeNet | CIFAR-10 |
| `python main.py app:yml/dorefa_resnet_cifar10.yml` | DoReFa ResNet-18 | CIFAR-10 |
| `python main.py app:yml/lenet_mnist.yml` | XNOR LeNet-5 | MNIST |
| `python main.py app:yml/mlp_mnist.yml` | XNOR MLP | MNIST |

Checkpoints are written to the path specified by `checkpoint` in the YAML file (default: `results/<run_name>/`).

---

## Repository Structure

```
.
├── main.py                      # Entry point: loads config, builds model, runs training
├── config.py                    # YAML → AttrDict config loader (FLAGS singleton)
├── requirements.txt
│
├── models/                      # CNN model definitions
│   ├── __init__.py              # Re-exports all model factory functions
│   ├── bnn_layers.py            # BNN binary layers (Binarize, BNNLinear, BNNConv2d)
│   ├── bnn_caffenet.py          # BNN CaffeNet for CIFAR-10
│   ├── xnor_layers.py           # XNOR binary layers (BinActive, XNORConv2d, XNORLinear, …)
│   ├── xnor_lenet.py            # XNOR LeNet-5 for MNIST
│   ├── xnor_mlp.py              # XNOR MLP for MNIST
│   ├── xnor_nin.py              # XNOR Network-in-Network for CIFAR-10
│   ├── dorefa_layers.py         # DoReFa quantization layers (DOREFAConv2d, DOREFALinear)
│   └── dorefa_resnet.py         # DoReFa ResNet-18/34/50/101/152 for CIFAR-10
│
├── classifiers/                 # Training and evaluation loops
│   ├── bnn_classifier.py        # BnnClassifier  (train_step, test, save_checkpoint)
│   ├── xnor_classifier.py       # XnorClassifier (train_step, test, save_checkpoint)
│   └── dorefa_classifier.py     # DorefaClassifier (train_step, test, save_checkpoint)
│
├── dataloader/                  # Dataset wrappers
│   ├── __init__.py
│   ├── cifar10.py               # CIFAR-10 train/test DataLoaders
│   └── mnist.py                 # MNIST train/test DataLoaders
│
├── yml/                         # Training configuration files
│   ├── bnn_caffenet_cifar10.yml
│   ├── dorefa_resnet_cifar10.yml
│   ├── lenet_mnist.yml
│   ├── mlp_mnist.yml
│   └── nin_cifar10.yml
│
└── tests/                       # Unit tests (pytest)
    ├── __init__.py
    ├── test_bnn_layers.py       # Tests for BNN layers
    ├── test_xnor_layers.py      # Tests for XNOR layers
    ├── test_dorefa_layers.py    # Tests for DoReFa layers
    ├── test_models.py           # Tests for model architectures
    └── test_classifiers.py     # Tests for classifier training loops
```

---

## Models

### BNN CaffeNet (`models/bnn_caffenet.py`)

Three binary convolutional blocks followed by a binary fully-connected layer. Designed for CIFAR-10 (input `3×32×32`).

| Layer | Type | Output |
|---|---|---|
| BNNConv2d + BN + Hardtanh + MaxPool | Binary conv | `32×16×16` |
| BNNConv2d + BN + Hardtanh + MaxPool | Binary conv | `32×8×8` |
| BNNConv2d + BN + Hardtanh + MaxPool | Binary conv | `32×4×4` |
| Flatten + BN1d + Hardtanh | — | `512` |
| BNNLinear + BN1d + LogSoftmax | Binary FC | `num_classes` |

Factory: `bnn_caffenet(num_classes=10)`

---

### XNOR LeNet-5 (`models/xnor_lenet.py`)

Classic LeNet-5 with the second conv replaced by `XNORConv2d`. Designed for MNIST (input `1×28×28`).

| Layer | Type | Output |
|---|---|---|
| Conv2d + BN + ReLU + MaxPool | Standard conv | `20×12×12` |
| XNORConv2d + MaxPool | XNOR binary conv | `50×4×4` |
| Flatten → BNLinearReLU | Binary FC block | `500` |
| Linear | Standard FC | `num_classes` |

Factory: `lenet5(out_classes=10)`

---

### XNOR MLP (`models/xnor_mlp.py`)

Two-hidden-layer MLP with one binary `BNLinearReLU` layer. Designed for MNIST (input `1×28×28`, flattened to `784`).

| Layer | Type | Output |
|---|---|---|
| Flatten → Linear + BN + ReLU | Standard FC | `512` |
| BNLinearReLU | Binary FC block | `256` |
| BN + Linear | Standard FC | `num_classes` |

Factory: `mlp(out_classes=10)`

---

### XNOR Network-in-Network (`models/xnor_nin.py`)

NIN architecture using `BNConvReLU` blocks (BatchNorm → BinActive → XNORConv2d → ReLU). Designed for CIFAR-10 (input `3×32×32`).

| Stage | Type |
|---|---|
| Conv2d + BN + ReLU | Standard conv |
| 2× BNConvReLU (1×1) | Binary conv blocks |
| MaxPool | — |
| 3× BNConvReLU (5×5, 1×1, 1×1) | Binary conv blocks (dropout=0.5 on first) |
| AvgPool | — |
| 2× BNConvReLU (3×3, 1×1) | Binary conv blocks (dropout=0.5 on first) |
| BN + Conv2d + ReLU + AdaptiveAvgPool + Flatten | Classifier head |

Factory: `nin(out_classes=10)`

---

### DoReFa ResNet-18 (`models/dorefa_resnet.py`)

ResNet-18 with all conv and FC layers replaced by `DOREFAConv2d` / `DOREFALinear` for arbitrary `wbit`/`abit` quantization. Designed for CIFAR-10 (input `3×32×32`).

| Parameter | Description |
|---|---|
| `wbit` | Weight quantization bit-width (1 = binary) |
| `abit` | Activation quantization bit-width (1 = binary) |

Factory: `dorefa_resnet18(wbit=1, abit=1)`

Additional factory functions available: `ResNet34`, `ResNet50`, `ResNet101`, `ResNet152`.

---

## Binary Layer Implementations

### BNN Layers (`models/bnn_layers.py`)

| Symbol | Description |
|---|---|
| `Binarize(tensor, quant_mode)` | Quantizes to ±1. `'det'` = deterministic sign; `'bin'` = threshold at 0; default = stochastic |
| `BNNLinear` | Subclass of `nn.Linear`. Binarizes `weight_org` before each forward pass. Skips input binarization for the first layer (784 or 3072 inputs) |
| `BNNConv2d` | Subclass of `nn.Conv2d`. Binarizes `weight_org` before each forward pass. Skips input binarization for 3-channel (RGB) inputs |

### XNOR Layers (`models/xnor_layers.py`)

| Symbol | Description |
|---|---|
| `BinActive` | Straight-through estimator (STE): `forward` = `sign(x)`; `backward` = identity for `\|x\| < 1`, else 0 |
| `XNORConv2d` | Convolves with `sign(w) × mean(\|w\|)`. Maintains full-precision `fp_weights`; `update_gradient()` computes the XNOR gradient correction |
| `XNORLinear` | Linear equivalent of `XNORConv2d` |
| `BNConvReLU` | Composite block: `BatchNorm2d → BinActive → XNORConv2d → ReLU` (optional dropout) |
| `BNLinearReLU` | Composite block: `BatchNorm1d → BinActive → XNORLinear → ReLU` (optional dropout) |

### DoReFa Layers (`models/dorefa_layers.py`)

| Symbol | Description |
|---|---|
| `ScaleSigner` | `forward` = `sign(x) × mean(\|x\|)`; passthrough `backward` |
| `Quantizer` | `forward` = `round(x × (2ⁿ−1)) / (2ⁿ−1)`; passthrough `backward` |
| `dorefa_w(w, nbit_w)` | Weight quantization: 1-bit → `ScaleSigner`; n-bit → tanh + uniform quantization to [−1, 1] |
| `dorefa_a(x, nbit_a)` | Activation quantization: clamp `0.1×x` to [0, 1] then quantize |
| `DOREFAConv2d` | Subclass of `nn.Conv2d` applying `dorefa_w` / `dorefa_a` in `forward`. Full-precision path when `nbit ≥ 32` |
| `DOREFALinear` | Subclass of `nn.Linear` applying `dorefa_w` / `dorefa_a` in `forward` |

---

## Classifiers

All three classifiers share the same interface and differ only in the binary-specific gradient handling during `train_step`.

| Classifier | Module | Binary-specific training logic |
|---|---|---|
| `BnnClassifier` | `classifiers/bnn_classifier.py` | Restores `weight_org` before each optimizer step; clamps `weight_org` to [−1, 1] after |
| `XnorClassifier` | `classifiers/xnor_classifier.py` | Calls `m.update_gradient()` on every `XNORConv2d` module after `loss.backward()` |
| `DorefaClassifier` | `classifiers/dorefa_classifier.py` | Standard backward pass (quantization STE handles gradients automatically) |

**Common interface:**

```python
classifier = BnnClassifier(model, train_loader, test_loader, device)
classifier.train(criterion, optimizer, epochs=300, scheduler=scheduler, checkpoint="results/run")
accuracy = classifier.test(criterion)   # returns top-1 % on the test set
```

Checkpoints are saved as `<checkpoint>_checkpoint.pth.tar` (every epoch) and `<checkpoint>_best.pth.tar` (best accuracy so far).

---

## Datasets

| Module | Dataset | Default input size | Auto-downloaded |
|---|---|---|---|
| `dataloader/cifar10.py` | CIFAR-10 | `3×32×32` | Yes (`datasets/cifar10/`) |
| `dataloader/mnist.py` | MNIST | `1×28×28` | Yes (`datasets/mnist/`) |

---

## Configuration Files

All YAML files live in `yml/` and are passed to `main.py` via `app:<path>`. Available keys:

| Key | Type | Description |
|---|---|---|
| `bin_type` | `str` | Binarization method: `'bnn'`, `'xnor'`, or `'dorefa'` |
| `model` | `str` | Model factory name: `bnn_caffenet`, `lenet5`, `mlp`, `nin`, `dorefa_resnet18` |
| `dataset` | `str` | Dataset module name: `'cifar10'` or `'mnist'` |
| `batch_size` | `int` | Training batch size |
| `test_batch_size` | `int` | Evaluation batch size |
| `optimizer` | `str` | `'adam'` or `'sgd'` |
| `lr` | `float` | Initial learning rate |
| `steps` | `list[int]` | Epochs at which the LR is multiplied by `gamma` |
| `gamma` | `float` | LR decay factor |
| `epochs` | `int` | Total training epochs |
| `checkpoint` | `str` | Path prefix for saving checkpoints |
| `no_cuda` | `bool` | Disable GPU even if available |

**Example** (`yml/nin_cifar10.yml`):

```yaml
bin_type: 'xnor'
model: "nin"
dataset: "cifar10"
batch_size: 128
test_batch_size: 100
optimizer: 'adam'
lr: 0.01
gamma: 0.1
steps: [80, 150]
epochs: 300
checkpoint: "results/nin_cifar10"
no_cuda: False
```

---

## Running the Tests

The `tests/` directory contains 110 unit tests covering layers, models, and classifiers. No datasets are downloaded — all tests use synthetic in-memory tensors.

```sh
pip install pytest
python -m pytest tests/ -v
```

| Test module | What is tested |
|---|---|
| `tests/test_bnn_layers.py` | `Binarize` (all modes, dtype preservation), `BNNLinear`, `BNNConv2d` |
| `tests/test_xnor_layers.py` | `BinActive` (STE forward/backward), `XNORConv2d`, `XNORLinear`, `BNConvReLU`, `BNLinearReLU` |
| `tests/test_dorefa_layers.py` | `ScaleSigner`, `Quantizer`, `dorefa_w`, `dorefa_a`, `DOREFAConv2d`, `DOREFALinear` |
| `tests/test_models.py` | Forward pass + output shape, `init_w`, NaN-free output for all five architectures |
| `tests/test_classifiers.py` | `train_step`, `test`, `save_checkpoint`, end-to-end one-epoch training for all three classifiers |

---

## Related Applications

If you find this code useful in your research, please consider citing:

- [Fast and Accurate Inference on Microcontrollers With Boosted Cooperative Convolutional Neural Networks (BC-Net)](https://ieeexplore.ieee.org/abstract/document/9275360)
- [CoopNet: Cooperative Convolutional Neural Network for Low-Power MCUs](https://ieeexplore.ieee.org/abstract/document/8964993)
- [TentacleNet: A Pseudo-Ensemble Template for Accurate Binary Convolutional Neural Networks](https://ieeexplore.ieee.org/abstract/document/9073982/)

---

## License

MIT
