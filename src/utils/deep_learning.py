"""A set of utilities for the MNIST example codebase."""

import os
from typing import Literal, Optional
import comet_ml
import numpy as np
import pydantic
import datasets
from tqdm import tqdm
import jax.numpy as jnp
import jax.random
import torchvision.transforms
import torch.utils.data

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_CACHE_DIR = os.path.join(PROJECT_DIR, "data")
IMAGE_MEAN = 0.130660
IMAGE_STD = 0.308108


class Metadata(pydantic.BaseModel):
    mode: Literal["flax", "pytorch", "preprocess-images"]
    dry_run: bool = pydantic.Field(..., description="only run a few samples")
    num_epochs: int = pydantic.Field(..., ge=1, le=10000)
    batch_size_training: int = pydantic.Field(..., ge=1, le=1024)
    batch_size_validation: int = pydantic.Field(..., ge=1, le=1024)
    learning_rate: float = pydantic.Field(..., ge=0.000001, le=1.0)
    momentum: float = pydantic.Field(..., ge=0.0, le=1.0)


def init_comet_experiment(metadata: Metadata) -> Optional[comet_ml.Experiment]:
    COMETML_API_KEY = os.getenv("COMETML_API_KEY")
    COMETML_PROJECT_NAME = os.getenv("COMETML_PROJECT_NAME")
    COMETML_WORKSPACE = os.getenv("COMETML_WORKSPACE")
    if all([COMETML_API_KEY, COMETML_PROJECT_NAME, COMETML_WORKSPACE]):
        print("initializing comet.ml experiment")
        experiment = comet_ml.Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name="flax-mnist",
            workspace="dostuffthatmatters",
        )
        experiment.log_parameters(metadata.dict())
        return experiment

    print("skip comet.ml initialization")
    return None


class RNG_Provider:
    def __init__(self, seed: int):
        self._rng = jax.random.PRNGKey(seed)

    def get(self) -> jax.random.PRNGKeyArray:
        self._rng, rng = jax.random.split(self._rng)
        return rng


class MNIST_FlaxDataset:
    def __init__(
        self,
        variant: Literal["training", "validation"],
        limit: Optional[int] = None,
    ) -> None:
        print("loading data from HuggingFace Datasets...")
        dataset = datasets.load_dataset("mnist", cache_dir=DATA_CACHE_DIR)[
            "train" if variant == "training" else "test"
        ]
        assert isinstance(dataset, datasets.Dataset)

        print("transforming and parsing data ...")
        self.sample_count = (
            len(dataset) if (limit is None) else min(len(dataset), limit)
        )
        numpy_images = np.array(
            [
                np.array(dataset[i]["image"]).reshape((28, 28, 1))
                for i in tqdm(range(self.sample_count), desc="images")
            ]
        )
        assert numpy_images.shape == (self.sample_count, 28, 28, 1)
        self.images = (
            (jnp.array(numpy_images, dtype=jnp.float32) / 255.0) - IMAGE_MEAN
        ) / IMAGE_STD
        numpy_labels = np.array(
            [dataset[i]["label"] for i in tqdm(range(self.sample_count), desc="labels")]
        )
        assert numpy_labels.shape == (self.sample_count,)
        self.labels = jnp.array(numpy_labels, dtype=jnp.int8)

    def __getitem__(self, index: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.images[index], self.labels[index]

    def __len__(self) -> int:
        return self.sample_count


class MNIST_PytorchDataset:
    def __init__(
        self,
        variant: Literal["training", "validation"],
        limit: Optional[int] = None,
    ) -> None:
        print("loading data from HuggingFace Datasets...")
        dataset = datasets.load_dataset("mnist", cache_dir=DATA_CACHE_DIR)[
            "train" if variant == "training" else "test"
        ]
        assert isinstance(dataset, datasets.Dataset)

        print("transforming and parsing data ...")
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((IMAGE_MEAN,), (IMAGE_STD,)),
            ]
        )
        self.sample_count = (
            len(dataset) if (limit is None) else min(len(dataset), limit)
        )
        self.images = [
            dataset[i]["image"] for i in tqdm(range(self.sample_count), desc="images")
        ]
        self.labels: list[str] = [
            dataset[i]["label"] for i in tqdm(range(self.sample_count), desc="labels")
        ]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        return self.transform(self.images[index]), self.labels[index]

    def __len__(self) -> int:
        return self.sample_count


def get_metrics_step_indices(steps_in_epoch: int) -> list[int]:
    prints_in_epoch: int = 10
    if steps_in_epoch <= 10:
        prints_in_epoch = 1
    elif steps_in_epoch <= 20:
        prints_in_epoch = 3
    elif steps_in_epoch <= 50:
        prints_in_epoch = 5
    assert prints_in_epoch <= steps_in_epoch
    return [
        max(list(xs)) for xs in np.array_split(range(steps_in_epoch), prints_in_epoch)
    ]
