import os
from typing import Literal
import click
import dotenv
import numpy

from src import utils, experiments

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


@click.command(help="XLA demo scripts")
@click.argument("mode", type=click.Choice(["numpy", "pytorch", "jax"]), required=True)
@click.option("--matrix-size", default=5000)
def xla(
    mode: Literal["numpy", "pytorch", "jax"],
    matrix_size: int,
) -> None:
    print(f"matrix: {matrix_size} x {matrix_size}, 64-Bit float")
    if mode == "numpy":
        print("running experiment 'xla.numpy'")
        experiments.xla_numpy.run(matrix_size)
    elif mode == "pytorch":
        print("running experiment 'xla.pytorch'")
        experiments.xla_pytorch.run(matrix_size)
    elif mode == "jax":
        print("running experiment 'xla.jax'")
        experiments.xla_jax.run(matrix_size)


@click.command(help="JIT demo scripts")
def jit() -> None:
    experiments.jit_jax.run()


@click.command(help="SciPy demo scripts")
@click.argument("mode", type=click.Choice(["numpy", "jax"]), required=True)
@click.option("--array-length", default=10_000)
def scipy(
    mode: Literal["numpy", "jax"],
    array_length: int,
) -> None:
    timestamps = numpy.linspace(0, 100, int(array_length))
    data_from_sensor_a = (
        numpy.sin(timestamps)
        + 1
        + timestamps / 10
        + numpy.random.normal(0, 0.3, int(array_length))
    )
    # a = sine wave + constant offset + linear slope + noise

    actual_slope = 1.5
    actual_offset = 3
    data_from_sensor_b = data_from_sensor_a.copy()
    data_from_sensor_b *= numpy.random.normal(actual_slope, 0.1, int(array_length))
    data_from_sensor_b += numpy.random.normal(actual_offset, 0.1, int(array_length))

    # expect: b = (a * slope) + offset
    # expect: a = (b - offset) / slope

    if mode == "numpy":
        print("running experiment 'scipy.numpy'")
        experiments.scipy_numpy.run(
            data_from_sensor_a,
            data_from_sensor_b,
            actual_slope,
            actual_offset,
        )
    elif mode == "jax":
        print("running experiment 'scipy.jax'")
        experiments.scipy_jax.run(
            data_from_sensor_a,
            data_from_sensor_b,
            actual_slope,
            actual_offset,
        )


@click.command()
@click.argument("mode", type=click.Choice(["pytorch", "jax"]), required=True)
def autograd(
    mode: Literal["pytorch", "jax"],
) -> None:
    if mode == "pytorch":
        print("running experiment 'autograd.pytorch'")
        experiments.autograd_pytorch.run()
    elif mode == "jax":
        print("running experiment 'autograd.jax'")
        experiments.autograd_jax.run()


@click.command(help="MNIST training and evaluation loop")
@click.argument(
    "mode", type=click.Choice(["flax", "pytorch", "preprocess-images"]), required=True
)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--num-epochs", default=10)
@click.option("--batch-size-training", default=128)
@click.option("--batch-size-validation", default=128)
@click.option("--learning-rate", default=0.01)
@click.option("--momentum", default=0.9)
def mnist(
    mode: Literal["flax", "pytorch", "preprocess-images"],
    dry_run: bool,
    num_epochs: int,
    batch_size_training: int,
    batch_size_validation: int,
    learning_rate: float,
    momentum: float,
) -> None:
    metadata = utils.deep_learning.Metadata(
        mode=mode,
        dry_run=dry_run,
        num_epochs=num_epochs,
        batch_size_training=batch_size_training,
        batch_size_validation=batch_size_validation,
        learning_rate=learning_rate,
        momentum=momentum,
    )
    print(f"metadata: {metadata}")

    if mode == "flax":
        print("running experiment 'mnist.flax'")
        experiment = utils.deep_learning.init_comet_experiment(metadata)
        experiments.mnist_flax.run_training(metadata, experiment)
    elif mode == "pytorch":
        print("running experiment 'mnist.pytorch'")
        experiments.mnist_pytorch.run_training(metadata)
    elif mode == "preprocess-images":
        print("running experiment 'mnist.preprocess-images'")
        utils.deep_learning.MNIST_FlaxDataset.compute_mean_and_std()


@click.group()
def cli() -> None:
    pass


cli.add_command(xla)
cli.add_command(jit)
cli.add_command(scipy)
cli.add_command(autograd)
cli.add_command(mnist)

if __name__ == "__main__":
    cli()
