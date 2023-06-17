import torch
from src import utils


def run(matrix_size: int) -> None:
    print(f"running xla_pytorch.py with matrix size: {matrix_size}")
    print(f"warning: the printouts will be coarsened to size 6")

    # set up gpu device

    device = torch.device("cuda:0")

    # generate random matrices

    with utils.timing.timed_section("generate random matrix"):
        a = torch.rand(matrix_size, matrix_size, device=device)
        b = torch.rand(matrix_size, matrix_size, device=device)

    assert a.device == device, f"a.device = {a.device}"
    assert b.device == device, f"b.device = {b.device}"
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
        c = torch.matmul(a, b)
        assert c.device == device, f"c.device = {c.device}"

    print("c =", utils.linalg.coarsen(a))

    # compute mean and sum

    mean_c = float(c.mean())
    sum_c = float(c.sum())
    expected_mean = 0.5 * 0.5 * matrix_size
    expected_sum = matrix_size * matrix_size * expected_mean
    utils.assertions.assert_similar_result("mean", expected_mean, mean_c)
    utils.assertions.assert_similar_result("sum", expected_sum, sum_c)

    # compute inverse

    with utils.timing.timed_section("compute inverse"):
        cinv = torch.inverse(c)
        assert cinv.device == device, f"cinv.device = {cinv.device}"

    # expect c * cinv == identity

    cid = torch.matmul(c, cinv)
    assert cid.device == device, f"cid.device = {cid.device}"
    print("cid = ", utils.linalg.coarsen(cid))
    utils.assertions.assert_similar_result(
        "mean of inverse * matrix",
        matrix_size,
        float(cid.sum()),
    )
