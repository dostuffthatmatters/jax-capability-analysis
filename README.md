# JAX Capability Analysis

This repository contains the demo project for my analysis of the capabilities of the Python library **JAX**.

<br/>

## 🛠️ Installation

This project only works on **Linux x86_64** systems with a **CUDA compatible GPU** and **CUDA 12.0** and **CuDNN 8.8.1** installed. The project has been tested on Ubuntu 20.04 with an Nvidia GeForce RTX 3060 Ti hosted on [Genesis Cloud](https://genesiscloud.com/).

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

The raw outputs of the following commands are included in this repository in the `logs/` directory: `./cli.sh xla numpy`, `./cli.sh xla pytorch`, `./cli.sh xla jax`, `./cli.sh jit`, `./cli.sh scipy numpy`, `./cli.sh scipy jax`, `./cli.sh autograd pytorch`, `./cli.sh autograd jax`, `./cli.sh mnist pytorch`, `./cli.sh mnist flax`.

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
