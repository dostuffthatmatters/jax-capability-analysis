# JAX Capability Analysis

This repository contains the demo project for my analysis of the capabilities of the Python library **JAX**.

<br/>

## 🛠️ Installation

This project only works on **Linux x86_64** systems with a **CUDA compatible GPU** and **CUDA 12.0** and **CuDNN 8.1** installed. The project has been tested on Ubuntu 20.04 with an Nvidia GeForce RTX 3060 Ti hosted on [Genesis Cloud](https://genesiscloud.com/).

Install the Python 3.11 project using the following steps:

```bash
# create a virtual environment
python3.11 -m venv .venv

# activate the virtual environment
source .venv/bin/activate

# install the dependencies
poetry install
```

Test the installation/codebase with the following commands:

```bash
# Test whether PyTorch and JAX can connect to the CUDA GPU
pytest -m installation tests/

# (developers only) test static type annotations with MyPy
pytest -m typing tests/
```

<br/>

## 🌋 Run the Analysis Scripts

The analysis scripts have been structured as a CLI using Click (https://click.palletsprojects.com/en/8.1.x/). The following commands show how to run the different experiments.

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

⚠️ After calling the CLI, it might take a few seconds until the experiment starts. This is mainly because the PyTorch library is big and must be loaded into memory.

<br/>

## 🐥 Improvements for the Development Experience of JAX and Flax

### Better Type information

Whenever deriving a JIT-compiled function or a gradient function from any function, the wrapped function loses all type information. This is a problem for static type checkers like MyPy, but most importantly for developers who want to understand their codebase by easily knowing the types of their variables.

The following code snippet implements a function `typed_jit` that does the same thing as `jit`, but preserves the type information of the wrapped function.

```python
import jax
import typing
import pydantic

T = typing.TypeVar("T", bound=typing.Callable)


def typed_jit(fun: T) -> T:
    return typing.cast(T, jax.jit(fun))


def f(x: jax.Array) -> jax.Array:
    """some docstring"""
    return 0.2 * x + jax.numpy.sin(x) + jax.numpy.cos(2 * x)


# has no type information
f_jit = jax.jit(f)

# preserves type information
f_jit_typed = typed_jit(f)
```

In the following demo video, you see that all typing information is preserved:

https://github.com/dostuffthatmatters/jax-capability-analysis/assets/29046316/16359d1a-54e6-43ea-8d2c-0e7755b5ab79

The same type of enhancement can be done with `jax.grad`. We can add all keyword arguments that `jax.grad` supports to the `typed_grad` function as well. However, I would like this to be implemented in the JAX library itself:

```python
def typed_grad(
    fun: T,
    argnums: int | typing.Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: typing.Sequence[typing.Hashable] = (),
) -> T:
    return typing.cast(
        T,
        jax.grad(
            fun,
            argnums=argnums,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            reduce_axes=reduce_axes,
        ),
    )


f_grad = jax.grad(f)
f_grad_typed = typed_grad(f)
```

<br/>

### PRNG Key Management

The way the JAX documentation wants users to generate new random keys is very cumbersome to read and write.

```python
rng = jax.random.PRNGKey(seed=42)

key1, rng = jax.random.split(rng)
function_that_uses_a_key(key1)

key2, rng = jax.random.split(rng)
function_that_uses_a_key(key2)
```

Using a small utility class, can make the code way more readable and does not lead to naming issues:

```python
class RNG_Provider:
    def __init__(self, seed: int):
        self._rng = jax.random.PRNGKey(seed)

    def get(self) -> jax.random.PRNGKeyArray:
        """split the current key and return one part of it"""
        self._rng, rng = jax.random.split(self._rng)
        return rng


rng_provider = RNG_Provider(seed=42)
function_that_uses_a_key(rng_provider.get())
function_that_uses_a_key(rng_provider.get())
```

I know this is not in the mind of JAX's design philosophy of forcing users to think about their PRNG usage, but I do not think this is hiding away any complexity that could lead to invalid results.

<br/>

### Metadata Management

In the examples of the Flax library, they use the `ConfigDict` object from Google's `ml_collections` library to manage Metadata: https://github.com/google/ml_collections#configdict. However, this is dynamically typed; hence cannot be type-checked and does not provide any annotations of help for the developer. You can look at https://github.com/google/flax/tree/main/examples/mnist and try to find the configurable options - without looking at the README because code should be self-documenting.

However, managing a statically typed dictionary with some validation rules is not very hard to implement. The following code snippet shows how to do this with the `pydantic` library:

```python
import pydantic

class Metadata(pydantic.BaseModel):
    mode: Literal["flax", "pytorch"]
    dry_run: bool = pydantic.Field(..., description="only run a few samples")
    num_epochs: int = pydantic.Field(..., ge=1, le=10000)
    batch_size_training: int = pydantic.Field(..., ge=1, le=1024)
    batch_size_validation: int = pydantic.Field(..., ge=1, le=1024)
    learning_rate: float = pydantic.Field(..., ge=0.000001, le=1.0)
    momentum: float = pydantic.Field(..., ge=0.0, le=1.0)


# statically typed metadata object can now be passed around
metadata = Metadata(
    mode="flax",
    dry_run=False,
    num_epochs=10,
    batch_size_training=64,
    batch_size_validation=64,
    learning_rate=0.001,
    momentum=0.9,
)
```

<br/>

### Improve The Development Experience of Your Deep Learning Stack

Instead of using the `ml_collections` [^1] library, I prefer using `pydantic` [^2] for validation and `click` [^3] for building a CLI - have a look at `main.py`. To make the dataset loading framework-agnostic, I like the `datasets` [^4] library by Hugging Face - then you will not have to load a huge PyTorch or TensorFlow dependency package into your project's virtual environment.

Deep Learning dependencies are very flaky because many of them depend on an exact version of, e.g. CUDA or CuDNN and have a lot of active contributors and frequent releases. Therefore you should not only include a `requirements.txt` file in your project with `pytorch==^1.8.1` but specify exact minor versions of the libraries you are using. I recommend using Poetry [^5] for documenting Python dependencies.

Static type checkers are very useful because they save you much debugging time at runtime and force you to have good function, class and variable annotations. When using MyPy [^6] to statically check your code, it will also parse your virtual environment; hence a big dependency like PyTorch or Tensorflow will add minutes to the initial run of MyPy (whenever it does not find a `.mypy_cache` directory).

[^1]: ML Collections https://github.com/google/ml_collections
[^2]: Pydantic https://github.com/pydantic/pydantic
[^3]: Click https://github.com/pallets/click
[^4]: Datasets https://github.com/huggingface/datasets
[^5]: Poetry https://github.com/python-poetry/poetry
[^6]: MyPy https://github.com/python/mypy

<br/>

## Raw Output from Running the Experiments

### XLA

**`./cli.sh xla numpy`:**

```
matrix size = 5000, 64-Bit float
Time to generate random matrix: 0.810971s
Time to compute mean and sum: 0.070553s
Time to compute product: 4.869536s
Time to compute inverse: 7.654785s
max abs offset from identity: 3.1141098588705063e-09
```

<br/>

**`./cli.sh xla pytorch`:**

```
matrix size = 5000, 64-Bit float
Time to generate random matrix: 0.141182s
Time to compute mean and sum: 0.019548s
Time to compute product: 0.026876s
Time to compute inverse: 1.669771s
max abs offset from identity: tensor(1.2526e-09, device='cuda:0', dtype=torch.float64)
```

<br/>

**`./cli.sh xla jax`:**

```
matrix size = 5000, 64-Bit float
Time to generate random matrix: 0.159070s
Time to compute mean and sum: 0.302824s
Time to compute product: 1.221243s
Time to compute inverse: 0.436647s
max abs offset from identity: 6.400581607035885e-10
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
parsing metadata
skip comet.ml initialization
running flax training and evaluation loop
Local devices detected by JAX: [StreamExecutorGpuDevice(id=0, process_index=0, slice_index=0)]
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

                                           CNN Summary
┏━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ path      ┃ module  ┃ inputs               ┃ outputs             ┃ params                     ┃
┡━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│           │ CNN     │ - float32[1,28,28,1] │ float32[1,10]       │                            │
│           │         │ - training: False    │                     │                            │
├───────────┼─────────┼──────────────────────┼─────────────────────┼────────────────────────────┤
│ Conv_0    │ Conv    │ float32[1,28,28,1]   │ float32[1,28,28,32] │ bias: float32[32]          │
│           │         │                      │                     │ kernel: float32[3,3,1,32]  │
│           │         │                      │                     │                            │
│           │         │                      │                     │ 320 (1.3 KB)               │
├───────────┼─────────┼──────────────────────┼─────────────────────┼────────────────────────────┤
│ Conv_1    │ Conv    │ float32[1,28,28,32]  │ float32[1,28,28,64] │ bias: float32[64]          │
│           │         │                      │                     │ kernel: float32[3,3,32,64] │
│           │         │                      │                     │                            │
│           │         │                      │                     │ 18,496 (74.0 KB)           │
├───────────┼─────────┼──────────────────────┼─────────────────────┼────────────────────────────┤
│ Dropout_0 │ Dropout │ float32[1,14,14,64]  │ float32[1,14,14,64] │                            │
├───────────┼─────────┼──────────────────────┼─────────────────────┼────────────────────────────┤
│ Dense_0   │ Dense   │ float32[1,12544]     │ float32[1,256]      │ bias: float32[256]         │
│           │         │                      │                     │ kernel: float32[12544,256] │
│           │         │                      │                     │                            │
│           │         │                      │                     │ 3,211,520 (12.8 MB)        │
├───────────┼─────────┼──────────────────────┼─────────────────────┼────────────────────────────┤
│ Dropout_1 │ Dropout │ float32[1,256]       │ float32[1,256]      │                            │
├───────────┼─────────┼──────────────────────┼─────────────────────┼────────────────────────────┤
│ Dense_1   │ Dense   │ float32[1,256]       │ float32[1,10]       │ bias: float32[10]          │
│           │         │                      │                     │ kernel: float32[256,10]    │
│           │         │                      │                     │                            │
│           │         │                      │                     │ 2,570 (10.3 KB)            │
├───────────┼─────────┼──────────────────────┼─────────────────────┼────────────────────────────┤
│           │         │                      │               Total │ 3,232,906 (12.9 MB)        │
└───────────┴─────────┴──────────────────────┴─────────────────────┴────────────────────────────┘

                              Total Parameters: 3,232,906 (12.9 MB)

epoch:   1
  training:
    loss=0.8327  accuracy=73.04
    loss=0.3174  accuracy=89.98
    ...
    loss=0.0971  accuracy=97.10
    loss=0.1057  accuracy=96.94
  validation:
    loss=0.0539  accuracy=98.20
epoch:   2
  training:
    loss=0.0978  accuracy=97.09
    loss=0.0816  accuracy=97.52
    ...
    loss=0.0716  accuracy=97.96
    loss=0.0722  accuracy=97.72
  validation:
    loss=0.0409  accuracy=98.64
epoch:   3
  training:
    loss=0.0618  accuracy=98.24
    loss=0.0603  accuracy=98.20
    ...
```
