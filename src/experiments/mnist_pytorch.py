# example derived from https://github.com/pytorch/examples/blob/main/mnist/main.py

from __future__ import print_function
from typing import Any

import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils.data

from src import utils


class CNN(torch.nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1
        )
        self.dropout1 = torch.nn.Dropout(p=0.25)
        self.fc1 = torch.nn.Linear(12 * 12 * 64, 256)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(256, 10)

    def forward(self, x: Any) -> Any:
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        # 28 x 28 x 1 -> 26 x 26 x 32

        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        # 26 x 26 x 32 -> 24 x 24 x 64

        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        x = self.dropout1(x)
        # 24 x 24 x 64 -> 12 x 12 x 64

        x = torch.flatten(x, start_dim=1)
        # 12 x 12 x 64 -> 9216

        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        # 9216 -> 256

        x = self.fc2(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        # 256 -> 10

        return x


def train(
    metadata: utils.deep_learning.Metadata,
    model: CNN,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader[Any],
    optimizer: torch.optim.SGD | torch.optim.Adadelta,
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
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()  # type: ignore
        optimizer.step()
        if batch_idx % 20 == 0:
            # compute accuracy
            pred = output.argmax(dim=1, keepdim=True)
            train_accuracy = pred.eq(target.view_as(pred)).sum().item() / len(data)
            print(
                f"Train Epoch: {epoch_index + 1} [{batch_idx * len(data):5d}/{sample_count:5d} "
                + f"({100.0 * batch_idx / len(dataloader):2.0f}%)] "
                + f"Loss: {loss.item():.6f}, Accuracy: {train_accuracy:.6f}"
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
            validation_loss += torch.nn.functional.nll_loss(
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


def run_training(
    metadata: utils.deep_learning.Metadata,
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
    optimizer = torch.optim.SGD(
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
