# example derived from https://github.com/pytorch/examples/blob/main/mnist/main.py

from __future__ import print_function
from typing import Any
import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data

from src import utils


# TODO: use same model as flax


class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(5 * 5 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: Any) -> Any:
        x = self.conv1(x)
        # 28 x 28 -> 26 x 26

        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=2)
        # 26 x 26 -> 13 x 13

        x = self.conv2(x)
        # 13 x 13 -> 11 x 11

        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=2)
        # 11 x 11 -> 5 x 5

        x = torch.flatten(x, start_dim=1)
        # 5 x 5 x 64 -> 1600

        x = self.fc1(x)
        x = F.relu(x)
        # 1600 -> 128

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(
    metadata: utils.deep_learning.Metadata,
    model: CNN,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader[Any],
    optimizer: optim.SGD | optim.Adadelta,
    epoch_index: int,
    sample_count: int,
) -> None:
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        assert list(data.shape)[1:] == [1, 28, 28]
        assert list(target.shape)[1:] == []

        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()  # type: ignore
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch_index + 1,
                    batch_idx * len(data),
                    sample_count,
                    100.0 * batch_idx / len(dataloader),
                    loss.item(),
                )
            )
            if metadata.dry_run:
                break


def validate(
    model: CNN,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader[Any],
    sample_count: int,
) -> None:
    model.eval()
    validation_loss: float = 0
    correct: int = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            assert list(data.shape)[1:] == [1, 28, 28]
            assert list(target.shape)[1:] == []

            output = model(data)
            validation_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= sample_count

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            validation_loss,
            correct,
            sample_count,
            100.0 * correct / sample_count,
        )
    )


# TODO: add comet ml integration


def run_training(
    metadata: utils.deep_learning.Metadata,
    experiment: comet_ml.Experiment,
) -> None:
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")

    torch.manual_seed(1)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Loading Training dataset")
    training_dataset = utils.deep_learning.MNIST_PytorchDataset(variant="training")
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,  # type: ignore
        batch_size=metadata.batch_size_training,
        num_workers=1 if use_cuda else 0,
        pin_memory=True if use_cuda else False,
        shuffle=True if use_cuda else None,
    )

    print("Loading Validation dataset")
    validation_dataset = utils.deep_learning.MNIST_PytorchDataset(variant="validation")
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,  # type: ignore
        batch_size=metadata.batch_size_validation,
        num_workers=1 if use_cuda else 0,
        pin_memory=True if use_cuda else False,
        shuffle=True if use_cuda else None,
    )

    cnn = CNN().to(device)
    optimizer = optim.SGD(
        cnn.parameters(),
        lr=metadata.learning_rate,
        momentum=metadata.momentum,
    )
    for epoch_index in range(metadata.num_epochs):
        train(
            metadata,
            cnn,
            device,
            training_dataloader,
            optimizer,
            epoch_index,
            sample_count=len(training_dataset),
        )
        validate(
            cnn,
            device,
            validation_dataloader,
            sample_count=len(validation_dataset),
        )

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")

    # TODO: export model
