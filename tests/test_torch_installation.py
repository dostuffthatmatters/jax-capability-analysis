import pytest
import torch


@pytest.mark.installation
def test_torch_installation() -> None:
    x = torch.rand(5, 3)
    print(x)

    assert torch.cuda.is_available()
    assert (
        torch.backends.cudnn.version() == 8801
    ), f"torch.backends.cudnn.version() = {torch.backends.cudnn.version()}"
