"""The OWL (PNW-Cnet v4) adapter. Importing this module loads TensorFlow."""

import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from robin_contracts.cards import AudioGeometry, ModelCard, RunnerResampled
from robin_contracts.inputs import AudioClip, Input
from robin_contracts.protocols import JsonScalar, Log, ModelContext
from robin_contracts.records import ClassScore, WindowOutput
from robin_contracts.specs import recipe, window_bounds

# Must be set before TensorFlow is imported: the weights are a Keras 2 file.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

_RUNTIME_MODULES = ("numpy", "soundfile", "tensorflow", "tf_keras", "sox_tensorflow")

try:
    import soundfile as sf
    import tensorflow as tf
    import tf_keras  # noqa: F401  Legacy Keras runs from it.
    from sox_tensorflow.processor import spectrogram_from_flac
except ModuleNotFoundError as error:
    if (error.name or "").split(".")[0] not in _RUNTIME_MODULES:
        raise
    raise ModuleNotFoundError(
        f"model owl/v4 needs {error.name}, which is not installed; "
        f"install robin-models[owl]",
        name=error.name,
    ) from error

# Batch size changes the scores at float32 rounding level, and nothing else.
DEFAULT_BATCH_SIZE = 64

# What OWL does whatever its card says. sox_tensorflow renders at 8 kHz from the first
# channel, resampled by soxr at its high-quality setting, and stretches a short last clip
# to the full image width. The network was trained on 12-second clips, outputs
# probabilities, and has no embedding output.
BEHAVIOUR = {
    "window_duration": 12.0,
    "sample_rate": 8000,
    "audio": AudioGeometry(
        downmix="first", resampler=RunnerResampled(algorithm="soxr_hq"), pad="time_scaled"
    ),
    "score_domain": "probability",
    "can_emit_embeddings": False,
}


def build(context: ModelContext) -> "OwlModel":
    """Check the resources and files, load the weights, and return the model."""
    card = context.card
    _refuse_a_card_owl_does_not_follow(card)
    batch_size = _read_batch_size(context.resources)
    if context.registry is None:
        raise ValueError("model owl/v4 needs a registry to name its outputs, and got none")
    weights = context.files.get("weights")
    if weights is None:
        raise ValueError(
            "model owl/v4 needs a file under the role 'weights', "
            f"and got the roles {sorted(context.files)}"
        )
    network = tf.keras.models.load_model(str(weights))
    _refuse_a_spectrogram_shape_the_network_does_not_take(card, network)
    return OwlModel(
        card=card,
        batch_size=batch_size,
        labels=tuple(entry.label for entry in context.registry.entries),
        network=network,
        scratch_dir=context.scratch_dir,
        log=context.log,
    )


def _refuse_a_card_owl_does_not_follow(card: ModelCard) -> None:
    for field, does in BEHAVIOUR.items():
        states = getattr(card, field)
        if states != does:
            raise ValueError(f"card field {field} is {states!r}, but model owl/v4 does {does!r}")


def _refuse_a_spectrogram_shape_the_network_does_not_take(card: ModelCard, network) -> None:
    # The network's input is (batch, height, width, channels).
    takes = tuple(network.input_shape[1:3])
    states = None if card.spectrogram_shape is None else tuple(card.spectrogram_shape)
    if states != takes:
        raise ValueError(
            f"card field spectrogram_shape is {states!r}, but the loaded network takes {takes!r}"
        )


def _read_batch_size(resources: Mapping[str, JsonScalar]) -> int:
    """The `batch_size` resource, or the default. Other resources are not OWL's to read."""
    size = resources.get("batch_size", DEFAULT_BATCH_SIZE)
    # bool is an int subclass, and True is a type error, not a batch of one.
    if type(size) is not int or size < 1:
        raise ValueError(f"resource batch_size must be a positive int, got {size!r}")
    return size


class OwlModel:
    """OWL and its loaded network, run over one recording at a time."""

    def __init__(
        self,
        *,
        card: ModelCard,
        batch_size: int,
        labels: tuple[str, ...],
        network: object,
        scratch_dir: Path,
        log: Log,
    ) -> None:
        self._card = card
        self._geometry = recipe(card).audio.geometry
        self._batch_size = batch_size
        self._labels = labels
        self._network = network
        self._scratch_dir = scratch_dir
        self._log = log
        self._png_dir: Path | None = None

    def run(self, input: Input) -> Iterator[WindowOutput]:
        """Render every window's spectrogram, then yield each window's scores in order."""
        if not isinstance(input, AudioClip):
            raise TypeError(f"model owl/v4 runs on audio, got a {type(input).__name__}")
        return self._windows(input.path)

    def _windows(self, audio: Path) -> Iterator[WindowOutput]:
        geometry = self._geometry
        bounds = window_bounds(sf.info(str(audio)).duration, geometry)
        # Set before rendering, so after_recording removes a partly rendered recording.
        self._png_dir = Path(tempfile.mkdtemp(prefix="owl-spectrograms-", dir=self._scratch_dir))
        self._log(f"rendering {len(bounds)} spectrograms for {audio.name}")
        paths = []
        for index, (start, _) in enumerate(bounds):
            dest = self._png_dir / f"{index}.png"
            # A render failure raises: skipping the window would leave a hole in the recording.
            spectrogram_from_flac(
                str(audio),
                start_time=start,
                duration=geometry.window_duration,
                shape=tuple(self._card.spectrogram_shape),
                dest=str(dest),
            )
            paths.append(str(dest))

        for index, scores in _predict(self._network, paths, self._batch_size):
            start, end = bounds[index]
            # Every label in output order, unfiltered: the engine keeps what was asked
            # for. strict refuses a network that gives the registry more or fewer scores.
            scored = tuple(
                ClassScore(label=label, score=score)
                for label, score in zip(self._labels, scores, strict=True)
            )
            yield WindowOutput(start=start, end=end, scores=scored)

    def after_recording(self) -> None:
        """Remove the recording's spectrograms. The network stays loaded."""
        if self._png_dir is not None:
            shutil.rmtree(self._png_dir, ignore_errors=True)
            self._png_dir = None

    def clean_up(self) -> None:
        """Nothing to release: the network stays loaded until the process ends."""


def _predict(
    network, paths: Sequence[str], batch_size: int
) -> Iterator[tuple[int, list[float]]]:
    """Yield `(window index, scores)` per spectrogram, matched by index, not position."""

    def load(path, index):
        image = tf.io.decode_png(tf.io.read_file(path), channels=1)
        return tf.cast(image, tf.float32) / 255.0, index

    dataset = (
        tf.data.Dataset.from_tensor_slices((list(paths), list(range(len(paths)))))
        .map(load, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
    )
    for images, indices in dataset:
        predictions = network(images).numpy()
        for index, vector in zip(indices.numpy(), predictions):
            yield int(index), vector.tolist()
