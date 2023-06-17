import jax.numpy
import jaxlib
import jax.lib


def test_jax_installation() -> None:
    assert jaxlib.__version__ == "0.4.10", f"jaxlib.__version__ = {jaxlib.__version__}"

    backend = jax.lib.xla_bridge.get_backend()

    assert backend.platform == "gpu", f"backend.platform = {backend.platform}"
    assert (
        backend.platform_version == "cuda 12000"
    ), f"backend.platform_version = {backend.platform_version}"
    assert (
        backend.device_count() >= 0
    ), f"backend.device_count() = {backend.device_count()}"

    x = jax.numpy.array([1, 2, 3]) / 3
    print(x)
