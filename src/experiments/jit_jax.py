import jax
import numpy

from src import utils

# TODO: add first print to main.py


def run() -> None:
    print("running jit_jax.py")

    # set up dummy function and gradient function

    def f(x: jax.Array) -> jax.Array:
        return 0.2 * x + jax.numpy.sin(x) + jax.numpy.cos(2 * x)

    # convert derived functions
    f_jit = jax.jit(f)
    f_jit_typed = utils.wrapper.typed_jit(f)

    # compute function and gradient for a single value

    x1 = jax.numpy.linspace(0, 4 * numpy.pi, 6, dtype=numpy.float16)
    y1 = f(x1)
    y1_jit = f_jit(x1)
    y1_jit_typed = f_jit_typed(x1)

    print("          x1 =", x1)
    print("          y1 =", y1)
    print("      y1_jit =", y1_jit)
    print("y1_jit_typed =", y1_jit_typed)

    # assert equal shapes
    assert y1.shape == x1.shape
    assert y1_jit.shape == x1.shape
    assert y1_jit_typed.shape == x1.shape

    # assert equal values
    assert jax.numpy.allclose(y1, y1_jit)
    assert jax.numpy.allclose(y1, y1_jit_typed)

    # time regular function

    computed_items = 100_000_000

    for array_length in [10_000, 100_000, 1_000_000, 10_000_000]:
        loop_count = computed_items // array_length
        print(f"array_length = {array_length}, loop_count = {loop_count}")

        print("  CPU:")
        jax.config.update("jax_platform_name", "cpu")  # type: ignore
        x1 = jax.numpy.linspace(0, 4 * numpy.pi, array_length, dtype=numpy.float32)
        with utils.timing.timed_section("compute normally", indent=4):
            for _ in range(loop_count):
                y1 = f(x1)
        y1 = f_jit_typed(x1)  # compile this outside of timing
        with utils.timing.timed_section("compute with JIT", indent=4):
            for _ in range(loop_count):
                y1 = f_jit_typed(x1)

        print("  GPU:")
        jax.config.update("jax_platform_name", "gpu")  # type: ignore
        x2 = jax.numpy.linspace(0, 4 * numpy.pi, array_length, dtype=numpy.float32)
        with utils.timing.timed_section("compute normally", indent=4):
            for _ in range(loop_count):
                y2 = f(x2)
        y2 = f_jit_typed(x2)  # compile this outside of timing
        with utils.timing.timed_section("compute with JIT", indent=4):
            for _ in range(loop_count):
                y2 = f_jit_typed(x2)
