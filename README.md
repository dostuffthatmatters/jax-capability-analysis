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

**`./cli.sh jit numpy`:**

```
TODO: add output here
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
TODO: add output here
```

<br/>

**`./cli.sh autograd jax`:**

```
TODO: add output here
```

<br/>

### MNIST

**`./cli.sh mnist pytorch`:**

```
TODO: add output here
```

<br/>

**`./cli.sh mnist flax`:**

```
TODO: add output here
```
