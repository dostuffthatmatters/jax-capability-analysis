import torch
import numpy

from src import utils


def run() -> None:
    print("running autograd_pytorch.py")

    # set up gpu device

    device = torch.device("cuda")

    # set up dummy function and gradient function

    def f(x: torch.Tensor) -> torch.Tensor:
        return 0.2 * x + torch.sin(x) + torch.cos(2 * x)

    def grad_f(x: torch.Tensor) -> torch.Tensor:
        _x = x.clone().detach().requires_grad_(True)
        f(_x).backward()  # type: ignore
        assert isinstance(_x.grad, torch.Tensor), "could not compute gradient"
        return _x.grad

    # compute function and gradient for a single value

    x1 = torch.tensor(3, dtype=torch.float32, device=device)
    y1 = f(x1)
    dy1 = grad_f(x1)
    assert dy1.shape == x1.shape

    print(f"x1 = {x1}")
    print(f"f(x1) = {y1}")
    print(f"grad_f(x1) = {dy1}")

    # compute function and gradient for a vector

    x2 = torch.linspace(0, 4 * numpy.pi, 20, dtype=torch.float32, device=device)
    y2 = f(x2)
    dy2 = torch.stack([grad_f(_x) for _x in x2])
    assert dy2.shape == x2.shape

    print(f"x2 = {x2}")
    print(f"f(x2) = {y2}")
    print(f"grad_f(x2) = {dy2}")

    # time the computation of the gradient on the GPU

    x3 = torch.linspace(0, 10 * numpy.pi, 1000, dtype=torch.float32, device=device)

    with utils.timing.timed_section(
        "perform gradient computation on GPU (1000 items, 10 times)"
    ):
        for _ in range(10):
            dy3 = torch.stack([grad_f(_x) for _x in x3])
            assert dy3.shape == x3.shape

    # time the computation of the gradient on the CPU

    x4 = torch.linspace(0, 10 * numpy.pi, 1000, dtype=torch.float32)

    with utils.timing.timed_section(
        "perform gradient computation on CPU (1000 items, 10 times)"
    ):
        for _ in range(10):
            dy4 = torch.stack([grad_f(_x) for _x in x4])
            assert dy4.shape == x4.shape
