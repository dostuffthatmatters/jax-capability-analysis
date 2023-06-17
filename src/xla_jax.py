from src import utils
import jax.random
import jax.numpy


def run(matrix_size: int) -> None:
    print(f"running xla_jax.py with matrix size: {matrix_size}")
    print(f"warning: the printouts will be coarsened to size 6")

    # generate random matrices

    key = jax.random.PRNGKey(0)
    subkey1, subkey2, key = jax.random.split(key, 3)

    with utils.timing.timed_section("generate random matrix"):
        a = jax.random.uniform(subkey1, (matrix_size, matrix_size))
        b = jax.random.uniform(subkey2, (matrix_size, matrix_size))

    print("a =", utils.linalg.coarsen(a))
    print("b =", utils.linalg.coarsen(b))

    # compute sum and means

    with utils.timing.timed_section("compute mean and sum"):
        mean_a = float(a.mean())
        mean_b = float(b.mean())
        sum_a = float(a.sum())
        sum_b = float(b.sum())

    # check matrix validities

    expected_mean = 0.5
    utils.assertions.assert_similar_result("mean", expected_mean, mean_a)
    utils.assertions.assert_similar_result("mean", expected_mean, mean_b)
    expected_sum = matrix_size * matrix_size * expected_mean
    utils.assertions.assert_similar_result("sum", expected_sum, sum_a)
    utils.assertions.assert_similar_result("sum", expected_sum, sum_b)

    # compute product

    with utils.timing.timed_section("compute product"):
        c = jax.numpy.matmul(a, b)

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
        cinv = jax.numpy.linalg.inv(c)

    # expect c * cinv == identity

    cid = jax.numpy.matmul(c, cinv)
    print("cid = ", utils.linalg.coarsen(cid))
    assert jax.numpy.allclose(cid, jax.numpy.eye(matrix_size)), "c * cinv != identity"
