import jax
import numpy

from src import utils

# TODO: add first print to main.py


def run() -> None:
    print("running autograd_jax.py")

    # set up dummy function and gradient function

    def f(x: jax.Array) -> jax.Array:
        return 0.2 * x + jax.numpy.sin(x) + jax.numpy.cos(2 * x)

    # convert derived functions
    f_jit = jax.jit(f)
    f_jit_typed = utils.wrapper.typed_jit(f)

    grad_f = jax.grad(f)
    grad_f_typed = utils.wrapper.typed_grad(f)
    grad_f_typed_jit = utils.wrapper.typed_jit(grad_f_typed)

    # compute function and gradient for a single value

    x1 = jax.numpy.array(3.0, dtype=numpy.float32)
    y1 = f(x1)
    dy1 = grad_f_typed(x1)
    assert dy1.shape == x1.shape

    print(f"x1 = {x1}")
    print(f"f(x1) = {y1}")
    print(f"grad_f(x1) = {dy1}")

    # compute function and gradient for a vector

    x2 = jax.numpy.linspace(0, 4 * numpy.pi, 20, dtype=numpy.float32)
    y2 = f(x2)
    dy2 = jax.vmap(grad_f_typed)(x2)
    assert dy2.shape == x2.shape

    print(f"x2 = {x2}")
    print(f"f(x2) = {y2}")
    print(f"grad_f(x2) = {dy2}")

    # time the computation of the gradient on the GPU

    x3 = jax.numpy.linspace(0, 10 * numpy.pi, 1000, dtype=numpy.float32)

    with utils.timing.timed_section(
        "perform gradient computation on GPU (1000 items, 10 times)"
    ):
        for _ in range(10):
            dy3 = jax.vmap(grad_f_typed)(x3)
            assert dy3.shape == x3.shape

    x3_jit = jax.numpy.linspace(0, 10 * numpy.pi, 1000, dtype=numpy.float32)

    with utils.timing.timed_section(
        "perform gradient computation on GPU with JIT (1000 items, 10 times)"
    ):
        for _ in range(10):
            dy3_jit = jax.vmap(grad_f_typed_jit)(x3_jit)
            assert dy3_jit.shape == x3_jit.shape

    # time the computation of the gradient on the CPU

    jax.config.update("jax_platform_name", "cpu")  # type: ignore
    x4 = jax.numpy.linspace(0, 10 * numpy.pi, 1000, dtype=numpy.float32)

    with utils.timing.timed_section(
        "perform gradient computation on CPU (1000 items, 10 times)"
    ):
        for _ in range(10):
            dy4 = jax.vmap(grad_f_typed)(x4)
            assert dy4.shape == x4.shape

    x4_jit = jax.numpy.linspace(0, 10 * numpy.pi, 1000, dtype=numpy.float32)

    with utils.timing.timed_section(
        "perform gradient computation on CPU with JIT (1000 items, 10 times)"
    ):
        for _ in range(10):
            dy4 = jax.vmap(grad_f_typed_jit)(x4_jit)
            assert dy4.shape == x4_jit.shape
