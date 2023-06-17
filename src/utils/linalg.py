from typing import Any, Literal
import jax
import numpy
import torch


def coarsen(
    matrix: numpy.ndarray[numpy.float32, Any] | jax.Array | torch.Tensor,
) -> numpy.ndarray[numpy.float32, Any]:
    """Coarsen a matrix down to a given new size (default is 6),
    by averaging over blocks of pixels. This averaging means,
    interpolating by nearest neighbor"""

    old_matrix = numpy.array(matrix)
    old_size = old_matrix.shape[0]
    new_size = 6
    assert old_matrix.shape[0] == old_matrix.shape[1]
    assert old_size >= new_size, f"old_size = {old_size}, new_size = {new_size}"

    new_matrix = numpy.zeros((new_size, new_size))

    for i in range(new_size):
        for j in range(new_size):
            new_matrix[i, j] = old_matrix[
                i * old_size // new_size : (i + 1) * old_size // new_size,
                j * old_size // new_size : (j + 1) * old_size // new_size,
            ].mean()

    return new_matrix
