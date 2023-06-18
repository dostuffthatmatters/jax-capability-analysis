import numpy
import numpy.linalg
import numpy.random
from src import utils


def run(matrix_size: int) -> None:
    print(f"matrix size = {matrix_size}, 64-Bit float")

    # generate random matrices

    with utils.timing.timed_section("generate random matrix"):
        a = numpy.random.rand(matrix_size, matrix_size).astype(numpy.float64)
        b = numpy.random.rand(matrix_size, matrix_size).astype(numpy.float64)

    # compute sum and means

    with utils.timing.timed_section("compute mean and sum"):
        mean_a = a.mean()
        mean_b = b.mean()
        sum_a = a.sum()
        sum_b = b.sum()

    # check matrix validities

    expected_mean = 0.5
    utils.assertions.assert_similar_result("mean", expected_mean, mean_a)
    utils.assertions.assert_similar_result("mean", expected_mean, mean_b)
    expected_sum = matrix_size * matrix_size * expected_mean
    utils.assertions.assert_similar_result("sum", expected_sum, sum_a)
    utils.assertions.assert_similar_result("sum", expected_sum, sum_b)

    # compute product

    with utils.timing.timed_section("compute product"):
        c = numpy.matmul(a, b).astype(numpy.float64)

    # compute mean and sum

    mean_c = c.mean()
    sum_c = c.sum()
    expected_mean = 0.5 * 0.5 * matrix_size
    expected_sum = matrix_size * matrix_size * expected_mean
    utils.assertions.assert_similar_result("mean", expected_mean, mean_c)
    utils.assertions.assert_similar_result("sum", expected_sum, sum_c)

    # compute inverse

    with utils.timing.timed_section("compute inverse"):
        cinv = numpy.linalg.inv(c)

    # expect c * cinv == identity

    # notice: the precision of 32-Bit inverse
    # computation with big matrices is terrible

    cid = numpy.matmul(c, cinv)
    print(
        f"max abs offset from identity:",
        numpy.abs(cid - numpy.eye(matrix_size, dtype=numpy.float64)).max(),
    )
