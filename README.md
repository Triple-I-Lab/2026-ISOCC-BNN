# An Area-Time-Efficient Pipelined Design for Binary Neural Networks

This repository accompanies the paper, **"An Area-Time-Efficient Pipelined Design for Binary Neural Networks."** It implements a binary neural network (BNN) for MNIST digit classification in PyTorch and deploys its exported 1-bit weights to a pipelined Verilog inference accelerator.

The design uses a `784-256-10` fully connected topology. Weights and activations are represented by one bit, so each dot product is implemented as XNOR plus popcount:

```text
w . x = 2 * popcount(XNOR(w, x)) - N
```

## Highlights

- **90.37% MNIST test accuracy** with the trained BNN included in this repository.
- **Two-stage pipelined neuron**: 16 parallel partial popcounts feed a second accumulation and activation stage.
- **Time-multiplexed first layer**: one pipelined neuron is reused over 256 hidden-layer outputs rather than instantiated 256 times.
- **BRAM-backed layer-one weights**, clock-enable gating, and a four-level tournament-tree argmax.
- Paper implementation results on a Xilinx Zynq UltraScale+: **4,234 LUTs**, **2,195 FFs**, **11 BRAMs**, **182 MHz**, **2.0 us** inference latency, and **0.606 W** power.

## Repository layout

```text
.
├── hardware_implementation/
│   ├── neuron_pipelined.v  # Reusable two-stage XNOR-popcount neuron
│   ├── top.v               # MNIST inference engine and control FSM
│   ├── tb_top.v            # Testbench for all 10,000 MNIST test images
│   └── constraints.xdc     # FPGA timing constraints
└── models/
    ├── models/bnn_mnist.py # 784-256-10 BNN definition
    ├── yml/bnn_mnist.yml   # Training configuration
    ├── convert_numpy.py    # Exports binary weights and test vectors for RTL
    ├── weights_layer*.txt  # Exported binary layer weights
    └── test_*.txt          # Exported MNIST test data and labels
```

## Model training

The `models/` directory is a modified clone of [lucamocerino/Binary-Neural-Networks-PyTorch-2.x](https://github.com/lucamocerino/Binary-Neural-Networks-PyTorch-2.x). It provides the PyTorch BNN training infrastructure used here and was adapted to train the `784-256-10` MNIST model deployed by this project. The hardware implementation in `hardware_implementation/` is specific to this work.

The trained model uses binarized weights and activations, no bias terms, and a `Hardtanh` hidden activation.

From `models/`, install the dependencies and train:

```sh
python -m pip install -r requirements.txt
python main.py app:yml/bnn_mnist.yml
```

The configuration trains for 300 epochs with SGD, a learning rate of `0.001`, batch size `128`, and learning-rate drops at epochs 80 and 150. Checkpoints are written beneath `models/results/`.

For the broader PyTorch BNN implementation, model variants, and tests, see [models/README.md](models/README.md).

## Export model data for RTL

The RTL consumes one-bit, text-formatted weights and MNIST vectors. After training, update the checkpoint path in `models/convert_numpy.py` if needed, then run:

```sh
cd models
python convert_numpy.py
```

This produces `weights_layer1.txt`, `weights_layer2.txt`, `test_data.txt`, and `test_labels.txt`, which are read by the Verilog design and testbench. The repository already includes exported data for the supplied 90.37% checkpoint.

## RTL inference accelerator

`hardware_implementation/top.v` accepts one 8-bit chunk of a binarized `28 x 28` image per clock. Assert `start` with the first byte, then provide the remaining 97 bytes on successive clock cycles. When inference completes, `done` pulses high and `classification` holds the predicted digit.

```text
start + 98 input bytes
        |
        v
  256 time-multiplexed layer-one evaluations
        |
        v
  10 parallel layer-two evaluations -> tournament-tree argmax -> done
```

The `neuron_pipelined` module splits each dot product into 16 groups. Stage 1 computes the group popcounts; Stage 2 sums them and calculates the signed BNN score. The layer-one weights are loaded with `$readmemb` into BRAM-oriented memory, while the smaller layer-two weights are held in registers.

### Simulate

An Icarus Verilog simulation can be run from the RTL directory:

```sh
cd hardware_implementation
iverilog -g2012 -o bnn_tb tb_top.v
vvp bnn_tb
```

The testbench streams all 10,000 included MNIST test examples and reports each prediction and the cumulative accuracy. Run it from `hardware_implementation/` so the relative paths to the files in `models/` resolve correctly.

### FPGA implementation

Open `hardware_implementation/top.v` and `hardware_implementation/constraints.xdc` in a Xilinx Vivado project for the target board. The supplied constraints define the clock and input timing; adjust the clock period and pin assignments for the selected FPGA platform.

## Results

The paper reports the following comparison on MNIST. `EqS` is an equivalent-slice area estimate, and area-time is measured in `EqS.us`.

| Implementation | Device | LUTs | FFs | BRAMs | Accuracy | Fmax | Latency | Images/s | Power | Area-time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| This work | Zynq UltraScale+ | 4,234 | 2,195 | 11 | 90.37% | 182 MHz | 2.0 us | 503,927 | 0.606 W | 1,626 |
| Ertorer and Unsalan | Artix-7 | 16,497 | 15,888 | 132 | 84.00% | 80 MHz | 17.8 us | 56,179 | 0.617 W | 97,384 |
| SFC-max (FINN) | ZC706 | 91,131 | - | 4.5 | 95.83% | 200 MHz | 0.31 us | 12,361,000 | 7.3 W | 3,567 |

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{attalla2026bnn,
  title={An Area-Time-Efficient Pipelined Design for Binary Neural Networks},
  author={Attalla, Joseph Yousry and Benton, Patriot and Aparicio, Tomas and Nguyen, Tuy Tan},
  year={2026}
}
```

## Acknowledgments

This work was supported by the Department of Electrical and Computer Engineering, FAMU-FSU College of Engineering, and the Undergraduate Research Opportunity Program at Florida State University.

The `models/` directory is based on [Binary-Neural-Networks-PyTorch-2.x](https://github.com/lucamocerino/Binary-Neural-Networks-PyTorch-2.x) by Luca Mocerino and has been modified for this project's MNIST BNN training workflow.
