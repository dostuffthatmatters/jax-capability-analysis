import numpy
from src import utils


def test_coarsen():
    a = numpy.random.rand(12, 12)
    b = utils.linalg.coarsen(a)
    assert b.shape == (6, 6)
    utils.assertions.assert_similar_result(
        "average of the whole matrix",
        float(a.mean()),
        float(b.mean()),
    )
    utils.assertions.assert_similar_result(
        "average of top left cell",
        float(a[0:2, 0:2].mean()),
        float(b[0, 0]),
    )
