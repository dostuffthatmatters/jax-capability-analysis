import math
from typing import Any, Optional
import comet_ml

import jax.numpy
import numpy

import flax.linen
import flax.training.train_state
import jax.nn
import optax

from src import utils


class CNN(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, x: Any, training: bool) -> Any:
        x = flax.linen.Conv(features=32, kernel_size=(3, 3))(x)
        x = flax.linen.relu(x)
        # 28 x 28 x 1 -> 26 x 26 x 32

        x = flax.linen.Conv(features=64, kernel_size=(3, 3))(x)
        x = flax.linen.relu(x)
        # 26 x 26 x 32 -> 24 x 24 x 64

        x = flax.linen.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # type: ignore
        x = flax.linen.Dropout(rate=0.25, deterministic=not training)(x)
        # 24 x 24 x 64 -> 12 x 12 x 64

        x = x.reshape((x.shape[0], -1))
        # 12 x 12 x 64 -> 9216

        x = flax.linen.Dense(features=256)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dropout(rate=0.5, deterministic=not training)(x)
        # 9216 -> 256

        x = flax.linen.Dense(features=10)(x)
        x = flax.linen.log_softmax(x, axis=1)
        # 256 -> 10

        return x


@jax.jit
def _apply_model(
    state: flax.training.train_state.TrainState,
    images: jax.Array,
    labels: jax.Array,
    dropout_key: Optional[jax.random.PRNGKeyArray] = None,
) -> tuple[Any, Any, Any]:
    """Computes gradients, loss and accuracy for a single batch."""

    sample_count = images.shape[0]
    assert images.shape == (sample_count, 28, 28, 1)
    assert labels.shape == (sample_count,)

    def loss_function(params: Any) -> tuple[Any, jax.numpy.ndarray]:
        if dropout_key is None:
            logits = state.apply_fn({"params": params}, images, training=False)
        else:
            logits = state.apply_fn(
                {"params": params},
                images,
                training=True,
                rngs={"dropout": dropout_key},
            )
        assert isinstance(logits, jax.numpy.ndarray)
        assert logits.shape == (sample_count, 10)

        one_hot = jax.nn.one_hot(labels, 10)
        assert one_hot.shape == (sample_count, 10)

        loss = jax.numpy.mean(
            optax.softmax_cross_entropy(logits=logits, labels=one_hot)
        )
        return loss, logits

    gradient_function = jax.value_and_grad(loss_function, has_aux=True)
    (loss, logits), grads = gradient_function(state.params)
    assert isinstance(logits, jax.numpy.ndarray)
    assert logits.shape == (sample_count, 10)

    accuracy = jax.numpy.mean(jax.numpy.argmax(logits, axis=-1) == labels)
    return grads, loss, accuracy


def _typed_apply_model(
    state: flax.training.train_state.TrainState,
    images: jax.Array,
    labels: jax.Array,
    dropout_key: Optional[jax.random.PRNGKeyArray] = None,
) -> tuple[Any, float, float]:
    """Same as _apply_model but with type annotations."""
    grads, loss, accuracy = _apply_model(
        state=state,
        images=images,
        labels=labels,
        dropout_key=dropout_key,
    )
    float_loss = float(loss)
    assert isinstance(float_loss, float)
    float_accuracy = float(accuracy)
    assert isinstance(float_accuracy, float)
    return grads, float_loss, float_accuracy


@jax.jit
def _update_model(
    state: flax.training.train_state.TrainState,
    grads: Any,
) -> flax.training.train_state.TrainState:
    return state.apply_gradients(grads=grads)  # type: ignore


def _train_one_epoch(
    metadata: utils.deep_learning.Metadata,
    state: flax.training.train_state.TrainState,
    dataset: utils.deep_learning.MNIST_FlaxDataset,
    experiment: Optional[comet_ml.Experiment],
    epoch_index: int,
    rng_provider: utils.deep_learning.RNG_Provider,
) -> flax.training.train_state.TrainState:
    """Train for a single epoch."""

    sample_count = len(dataset)
    steps_in_epoch = math.floor(sample_count / metadata.batch_size_training)

    perms = jax.random.permutation(rng_provider.get(), sample_count)
    assert isinstance(perms, jax.numpy.ndarray)
    assert perms.shape == (sample_count,)

    # skip incomplete batch at the end of the dataset
    perms = perms[: steps_in_epoch * metadata.batch_size_training].reshape(
        (steps_in_epoch, metadata.batch_size_training),
    )

    epoch_loss: list[float] = []
    epoch_acc: list[float] = []

    # indices at which to log metrics
    metrics_step_indices = utils.deep_learning.get_metrics_step_indices(steps_in_epoch)

    print(f"  training:")

    for step_index, perm in enumerate(perms):
        batch_images = dataset.images[perm, ...]
        batch_labels = dataset.labels[perm, ...]
        assert batch_images.shape == (metadata.batch_size_training, 28, 28, 1)
        assert batch_labels.shape == (metadata.batch_size_training,)

        grads, loss, accuracy = _typed_apply_model(
            state=state,
            images=batch_images,
            labels=batch_labels,
            dropout_key=rng_provider.get(),
        )
        epoch_loss.append(loss)
        epoch_acc.append(accuracy)

        state = _update_model(state, grads)

        if step_index in metrics_step_indices:
            mean_accuracy = float(numpy.mean(epoch_acc))
            mean_loss = float(numpy.mean(epoch_loss))
            assert isinstance(mean_accuracy, float)
            assert isinstance(mean_loss, float)

            epoch_acc.clear()
            epoch_loss.clear()

            print(f"    loss={mean_loss:.4f}  accuracy={(mean_accuracy * 100):.2f}")
            if experiment is not None:
                experiment.log_metric(
                    "training_accuracy",
                    mean_accuracy,
                    epoch=(epoch_index + 1),
                    step=(epoch_index * steps_in_epoch) + step_index + 1,
                )
                experiment.log_metric(
                    "training_loss",
                    mean_loss,
                    epoch=(epoch_index + 1),
                    step=(epoch_index * steps_in_epoch) + step_index + 1,
                )

    return state


def _validate_model(
    metadata: utils.deep_learning.Metadata,
    state: flax.training.train_state.TrainState,
    dataset: utils.deep_learning.MNIST_FlaxDataset,
    experiment: Optional[comet_ml.Experiment],
    epoch_index: int,
) -> tuple[float, float]:
    """Train for a single epoch. Returns tuple of (loss, accuracy)."""

    sample_count = len(dataset)
    image_batches: list[jax.Array] = []
    label_batches: list[jax.Array] = []

    number_of_chunks = math.ceil(sample_count / metadata.batch_size_validation)
    for i in range(number_of_chunks):
        start_index = i * metadata.batch_size_validation
        end_index = (i + 1) * metadata.batch_size_validation
        if i == (number_of_chunks - 1):
            end_index = sample_count - 1
        image_batches.append(dataset.images[start_index:end_index])
        label_batches.append(dataset.labels[start_index:end_index])

    assert len(image_batches) == number_of_chunks
    assert len(label_batches) == number_of_chunks

    validation_losses: list[float] = []
    validation_accuracies: list[float] = []

    print(f"  validation:")

    for batch_index in range(number_of_chunks):
        image_batch = image_batches[batch_index]
        label_batch = label_batches[batch_index]

        assert list(image_batch.shape)[1:] == [28, 28, 1]
        assert list(label_batch.shape)[1:] == []

        _, loss, accuracy = _typed_apply_model(
            state=state,
            images=image_batch,
            labels=label_batch,
        )
        validation_losses.append(loss)
        validation_accuracies.append(accuracy)

    mean_loss = float(numpy.mean(validation_losses))
    mean_accuracy = float(numpy.mean(validation_accuracies))
    assert isinstance(mean_accuracy, float)
    assert isinstance(mean_loss, float)

    print(f"    loss={mean_loss:.4f}  accuracy={(mean_accuracy * 100):.2f}")
    if experiment is not None:
        steps_per_training_epoch = math.floor(
            len(dataset) / metadata.batch_size_training
        )
        experiment.log_metric(
            "validation_loss",
            mean_loss,
            epoch=epoch_index + 1,
            step=(epoch_index + 1) * steps_per_training_epoch,
        )
        experiment.log_metric(
            "validation_accuracy",
            mean_accuracy,
            epoch=epoch_index + 1,
            step=(epoch_index + 1) * steps_per_training_epoch,
        )

    return mean_loss, mean_accuracy


def run_training(
    metadata: utils.deep_learning.Metadata,
    experiment: Optional[comet_ml.Experiment],
) -> None:
    """Run the training loop for an MNIST CNN model."""

    print(f"Local devices detected by JAX: {jax.local_devices()}")

    print("Loading Training dataset")
    training_dataset = utils.deep_learning.MNIST_FlaxDataset(
        variant="training", limit=1024 if metadata.dry_run else None
    )

    print("Loading Validation dataset")
    validation_dataset = utils.deep_learning.MNIST_FlaxDataset(
        variant="validation", limit=1024 if metadata.dry_run else None
    )

    # initialize Pseudo-Random Number Generator
    rng_provider = utils.deep_learning.RNG_Provider(seed=0)

    # initialize model
    cnn = CNN()  # type: ignore
    model_preview = cnn.tabulate(
        jax.random.PRNGKey(0),
        jax.numpy.ones((1, 28, 28, 1)),
        training=False,
    )
    if experiment is not None:
        experiment.log_text(model_preview)
    print(model_preview)

    # initialize optimizer and train state
    params = cnn.init(
        rng_provider.get(),
        jax.numpy.ones([1, 28, 28, 1]),
        training=False,
    )["params"]

    optimizer = optax.sgd(
        learning_rate=metadata.learning_rate,
        momentum=metadata.momentum,
    )
    state = flax.training.train_state.TrainState.create(  # type: ignore
        apply_fn=cnn.apply,
        params=params,
        tx=optimizer,
    )
    for epoch_index in range(metadata.num_epochs):
        print(f"epoch: {epoch_index+1:3d}")
        state = _train_one_epoch(
            metadata=metadata,
            state=state,
            dataset=training_dataset,
            experiment=experiment,
            epoch_index=epoch_index,
            rng_provider=rng_provider,
        )

        _validate_model(
            metadata=metadata,
            state=state,
            dataset=validation_dataset,
            experiment=experiment,
            epoch_index=epoch_index,
        )

    # TODO: add early stopping based on accuracy
    # TODO: export model
