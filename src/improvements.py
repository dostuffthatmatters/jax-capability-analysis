import jax
import typing
from typing import Literal
import pydantic

# TYPED JIT

T = typing.TypeVar("T", bound=typing.Callable)


def typed_jit(fun: T) -> T:
    return typing.cast(T, jax.jit(fun))


def f(x: jax.Array) -> jax.Array:
    return 0.2 * x + jax.numpy.sin(x) + jax.numpy.cos(2 * x)


f_jit = jax.jit(f)
f_jit_typed = typed_jit(f)

# TYPED GRAD


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

# RNG MANAGEMENT


class RNG_Provider:
    def __init__(self, seed: int):
        self._rng = jax.random.PRNGKey(seed)

    def get(self) -> jax.random.PRNGKeyArray:
        self._rng, rng = jax.random.split(self._rng)
        return rng


# METADATA PROVIDER


class Metadata(pydantic.BaseModel):
    mode: Literal["flax", "pytorch"]
    dry_run: bool = pydantic.Field(..., description="only run a few samples")
    num_epochs: int = pydantic.Field(..., ge=1, le=10000)
    batch_size_training: int = pydantic.Field(..., ge=1, le=1024)
    batch_size_validation: int = pydantic.Field(..., ge=1, le=1024)
    learning_rate: float = pydantic.Field(..., ge=0.000001, le=1.0)
    momentum: float = pydantic.Field(..., ge=0.0, le=1.0)


metadata = Metadata(
    mode="flax",
    dry_run=False,
    num_epochs=10,
    batch_size_training=64,
    batch_size_validation=64,
    learning_rate=0.001,
    momentum=0.9,
)
