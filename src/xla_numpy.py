import numpy
import numpy.linalg
import numpy.random
from src import utils


def run(matrix_size: int) -> None:
    print(f"running xla_numpy.py with matrix size: {matrix_size}")
    print(f"warning: the printouts will be coarsened to size 6")

    # generate random matrices

    with utils.timing.timed_section("generate random matrix"):
        a = numpy.random.rand(matrix_size, matrix_size)
        b = numpy.random.rand(matrix_size, matrix_size)

    print("a =", utils.linalg.coarsen(a))
    print("b =", utils.linalg.coarsen(b))

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
        c = numpy.matmul(a, b)

    print("c =", utils.linalg.coarsen(a))

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

    cid = numpy.matmul(c, cinv)
    print("cid = ", utils.linalg.coarsen(cid))
    assert numpy.allclose(cid, numpy.eye(matrix_size)), "c * cinv != identity"
