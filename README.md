# JAX Capability Analysis

This repository contains the demo project for my analysis of the capabilities of the Python library **JAX**.

<br/>

## 🛠️ Installation

This project only works on **Linux x86_64** systems that have a **CUDA compatible GPU** and **CUDA 12.0** as well as CuDNN 8.1 installed. The project has been tested on Ubuntu 20.04 with a NVIDIA GeForce RTX 3060 Ti hosted on [Genesis Cloud](https://genesiscloud.com/).

Install the Python 3.11 project using the following steps:

```bash
# create a virtual environment
python3.11 -m venv .venv

# activate the virtual environment
source .venv/bin/activate

# install the dependencies
poetry install
```

<br/>

## 🌋 Run the Analysis Scripts

The analysis scripts have been structured as a CLI using Click (https://click.palletsprojects.com/en/8.1.x/). The following commands will show you how to run the experiments of the different sections:

```bash
# show help menu of the whole project
python main.py --help

# show help menu of the individual sections
python main.py xla --help
python main.py autograd --help
python main.py scipy --help
python main.py mnist --help

# an example of how to run the XLA experiment for
# jax with a non-default matrix size
python main.py xla jax --matrix-size 1000
```

⚠️ After calling the CLI, it might take a few seconds until the experiment starts. This is mainly because the PyTorch library is very big and has to be loaded into memory.

<br/>

## 🐥 Improvements for the Development Experience

TODO: add improvements here

## Raw Output from Running the Experiments

### XLA

**`./cli.sh xla numpy`:**

```
TODO: add output here
```

<br/>

**`./cli.sh xla jax`:**

```
TODO: add output here
```

<br/>

**`./cli.sh xla pytorch`:**

```
TODO: add output here
```

<br/>

### JIT

**`./cli.sh jit`:**

```
running jit_jax.py
          x1 = [ 0.     2.512  5.023  7.535 10.05  12.56 ]
          y1 = [ 1.     1.398 -0.76   1.653  1.748  3.508]
      y1_jit = [ 1.     1.398 -0.76   1.653  1.748  3.508]
y1_jit_typed = [ 1.     1.398 -0.76   1.653  1.748  3.508]
array_length = 10000, loop_count = 10000
  CPU:
    Time to compute normally: 4.978520s
    Time to compute with JIT: 0.341038s
  GPU:
    Time to compute normally: 4.668422s
    Time to compute with JIT: 0.407004s
array_length = 100000, loop_count = 1000
  CPU:
    Time to compute normally: 0.996854s
    Time to compute with JIT: 0.036692s
  GPU:
    Time to compute normally: 0.449717s
    Time to compute with JIT: 0.037864s
array_length = 1000000, loop_count = 100
  CPU:
    Time to compute normally: 0.643250s
    Time to compute with JIT: 0.003452s
  GPU:
    Time to compute normally: 0.041690s
    Time to compute with JIT: 0.003868s
array_length = 10000000, loop_count = 10
  CPU:
    Time to compute normally: 0.581300s
    Time to compute with JIT: 0.000339s
  GPU:
    Time to compute normally: 0.007957s
    Time to compute with JIT: 0.002285s
```

<br/>

### SciPy

**`./cli.sh scipy numpy`:**

```
Time to run rmse_with_calibration (iteration #0): 0.064266s
Time to run rmse_with_calibration (iteration #1): 0.065531s
Time to run rmse_with_calibration (iteration #2): 0.065142s
Time to run rmse_with_calibration (iteration #3): 0.065827s
Time to run rmse_with_calibration (iteration #4): 0.071322s
Optimization terminated successfully.
         Current function value: 0.081186
         Iterations: 10
         Function evaluations: 51
         Gradient evaluations: 17
optimization_result:   message: Optimization terminated successfully.
  success: True
   status: 0
      fun: 0.0811855234492035
        x: [ 1.520e+00  3.000e+00]
      nit: 10
      jac: [-4.221e-04 -8.054e-04]
 hess_inv: [[ 4.230e-01 -2.967e-02]
            [-2.967e-02  1.973e-01]]
     nfev: 51
     njev: 17
```

<br/>

**`./cli.sh scipy jax`:**

```
Time to run rmse_with_calibration (iteration #0): 3.562486s
Time to run rmse_with_calibration (iteration #1): 0.000992s
Time to run rmse_with_calibration (iteration #2): 0.000501s
Time to run rmse_with_calibration (iteration #3): 0.000459s
Time to run rmse_with_calibration (iteration #4): 0.000395s
changing input shape for jitted function
Time to run rmse_with_calibration (iteration #5): 0.121439s
Time to run rmse_with_calibration (iteration #6): 0.000703s
Time to run rmse_with_calibration (iteration #7): 0.000552s
Time to run rmse_with_calibration (iteration #8): 0.000544s
Time to run rmse_with_calibration (iteration #9): 0.000486s
changing input shape for jitted function
Time to run rmse_with_calibration (iteration #10): 0.115021s
Time to run rmse_with_calibration (iteration #11): 0.000634s
Time to run rmse_with_calibration (iteration #12): 0.000432s
Time to run rmse_with_calibration (iteration #13): 0.000398s
Time to run rmse_with_calibration (iteration #14): 0.000502s
optimization_result: OptimizeResults(x=Array([1.5332441, 2.8007357], dtype=float32), success=Array(True, dtype=bool), status=Array(0, dtype=int32, weak_type=True), fun=Array(0.4449536, dtype=float32), jac=Array([-1.9930303e-06, -2.5134068e-07], dtype=float32), hess_inv=Array([[ 0.12239841, -0.74058217],
       [-0.74058217,  5.5074096 ]], dtype=float32), nfev=Array(13, dtype=int32, weak_type=True), njev=Array(13, dtype=int32, weak_type=True), nit=Array(12, dtype=int32, weak_type=True))
```

<br/>

### Autograd

**`./cli.sh autograd pytorch`:**

```
        x1 = 3.0
     f(x1) = 1.7012903690338135
grad_f(x1) = -0.23116151988506317

        x2 = tensor([ 0.0000,  0.6614,  1.3228,  1.9842,  2.6456,  3.3069,  3.9683,  4.6297,
         5.2911,  5.9525,  6.6139,  7.2753,  7.9367,  8.5980,  9.2594,  9.9208,
        10.5822, 11.2436, 11.9050, 12.5664], device='cuda:0')
     f(x2) = tensor([ 1.0000,  0.9920,  0.3545,  0.6353,  1.5520,  1.4426, -0.0246, -1.0570,
        -0.1806,  1.6549,  2.4366,  1.8905,  1.5976,  2.3728,  2.9623,  2.0552,
         0.5234,  0.3998,  2.0123,  3.5133], device='cuda:0')
grad_f(x2) = tensor([ 1.2000, -0.9497, -0.5064,  1.2698,  0.9949, -1.4358, -2.4705, -0.2118,
         2.5785,  2.3742, -0.0826, -1.0846,  0.4466,  1.5159, -0.1370, -2.3538,
        -1.6731,  1.3974,  2.9279,  1.2000], device='cuda:0')

Time to perform gradient computation on GPU (1000 items, 10 times): 3.962198s
Time to perform gradient computation on CPU (1000 items, 10 times): 1.690767s
```

<br/>

**`./cli.sh autograd jax`:**

```
        x1 = 3.0
     f(x1) = 1.7012903690338135
grad_f(x1) = -0.23116151988506317

        x2 = [ 0.         0.661388   1.322776   1.984164   2.645552   3.3069398
  3.968328   4.629716   5.291104   5.9524918  6.6138797  7.2752676
  7.936656   8.598043   9.259432   9.920819  10.582208  11.243596
 11.9049835 12.566371 ]
     f(x2) = [ 1.          0.9919757   0.35448164  0.63532484  1.552006    1.4426103
 -0.02463884 -1.0570028  -0.18064016  1.6549404   2.436616    1.8905241
  1.5975547   2.3727536   2.962298    2.0551636   0.523385    0.3998468
  2.012272    3.5132744 ]
grad_f(x2) = [ 1.2        -0.9496601  -0.50640905  1.2697529   0.9948586  -1.435761
 -2.4704502  -0.2117664   2.578496    2.3742416  -0.08260958 -1.0845982
  0.44661242  1.5158873  -0.1369658  -2.3538072  -1.6731402   1.3973863
  2.9279408   1.1999986 ]

Time to perform gradient computation on GPU w/o JIT  (1000 items, 10 times): 0.873759s
Time to perform gradient computation on GPU with JIT (1000 items, 10 times): 0.187574s
Time to perform gradient computation on CPU w/o JIT  (1000 items, 10 times): 0.098652s
Time to perform gradient computation on CPU with JIT (1000 items, 10 times): 0.007690s
```

<br/>

### MNIST

**`./cli.sh mnist pytorch`:**

```
parsing metadata
skip comet.ml initialization
running pytorch training and evaluation loop
Using CUDA: True
Loading Training dataset
loading data from HuggingFace Datasets...
Found cached dataset mnist (/home/ubuntu/Documents/jax-capability-analysis/data/mnist/mnist/1.0.0/9d494b7f466d6931c64fb39d58bb1249a4d85c9eb9865d9bc20960b999e2a332)
100%|█████████████████████████████████████████████████| 2/2 [00:00<00:00, 395.88it/s]
transforming and parsing data ...
images: 100%|█████████████████████████████████████████| 60000/60000 [00:09<00:00, 6640.24it/s]
labels: 100%|█████████████████████████████████████████| 60000/60000 [00:08<00:00, 7474.88it/s]
Loading Validation dataset
loading data from HuggingFace Datasets...
Found cached dataset mnist (/home/ubuntu/Documents/jax-capability-analysis/data/mnist/mnist/1.0.0/9d494b7f466d6931c64fb39d58bb1249a4d85c9eb9865d9bc20960b999e2a332)
100%|█████████████████████████████████████████████████| 2/2 [00:00<00:00, 481.63it/s]
transforming and parsing data ...
images: 100%|█████████████████████████████████████████| 10000/10000 [00:01<00:00, 7134.12it/s]
labels: 100%|█████████████████████████████████████████| 10000/10000 [00:01<00:00, 7391.99it/s]
Train Epoch: 1 [    0/60000 ( 0%)] Loss: 2.304506
Train Epoch: 1 [ 1280/60000 ( 2%)] Loss: 1.785377
Train Epoch: 1 [ 2560/60000 ( 4%)] Loss: 0.892751
...
Train Epoch: 1 [57600/60000 (96%)] Loss: 0.080127
Train Epoch: 1 [58880/60000 (98%)] Loss: 0.149030

Test set: Average loss: 0.0734, Accuracy: 9760/10000 (98%)

Train Epoch: 2 [    0/60000 ( 0%)] Loss: 0.056260
Train Epoch: 2 [ 1280/60000 ( 2%)] Loss: 0.137641
Train Epoch: 2 [ 2560/60000 ( 4%)] Loss: 0.060220
...
Train Epoch: 2 [57600/60000 (96%)] Loss: 0.097883
Train Epoch: 2 [58880/60000 (98%)] Loss: 0.172521

Test set: Average loss: 0.0480, Accuracy: 9843/10000 (98%)

Train Epoch: 3 [    0/60000 ( 0%)] Loss: 0.066899
Train Epoch: 3 [ 1280/60000 ( 2%)] Loss: 0.046682
Train Epoch: 3 [ 2560/60000 ( 4%)] Loss: 0.191935
...
```

<br/>

**`./cli.sh mnist flax`:**

```
TODO: add output here
```
