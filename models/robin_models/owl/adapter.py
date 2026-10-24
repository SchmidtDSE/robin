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
from robin_models._adapter_checks import (
    check_device,
    get_batch_size,
    refuse_a_card_the_model_does_not_follow,
)
from robin_models.registration import WEIGHTS_ROLE

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

OWNER = "model owl/v4"
# Batch size changes the scores at float32 rounding level, and nothing else.
DEFAULT_BATCH_SIZE = 64
TF_DEVICE = "/CPU:0"

# What OWL does whatever its card says. sox_tensorflow renders at 8 kHz from the first
# channel, resampled by soxr at its high-quality setting, and renders only full windows.
# The network was trained on 12-second clips, ends in a sigmoid for each class, and has no
# embedding output.
BEHAVIOUR = {
    "window_duration": 12.0,
    "sample_rate": 8000,
    "audio": AudioGeometry(
        downmix="first", resampler=RunnerResampled(algorithm="soxr_hq"), pad="drop"
    ),
    "score_domain": "sigmoid",
    "can_emit_embeddings": False,
}


def build(context: ModelContext) -> "OwlModel":
    """Check the card, resources and files, load the weights, and return the model."""
    card = context.card
    refuse_a_card_the_model_does_not_follow(card, BEHAVIOUR, OWNER)
    batch_size = get_batch_size(context.resources, DEFAULT_BATCH_SIZE)
    check_device(context.resources, OWNER)
    if context.registry is None:
        raise ValueError(f"{OWNER} needs a registry to name its outputs, and got none")
    weights = context.files.get(WEIGHTS_ROLE)
    if weights is None:
        raise ValueError(
            f"{OWNER} needs a file under the role {WEIGHTS_ROLE!r}, "
            f"and got the roles {sorted(context.files)}"
        )
    with tf.device(TF_DEVICE):
        network = tf.keras.models.load_model(str(weights))
    _refuse_a_spectrogram_shape_the_network_does_not_take(card, network)
    return OwlModel(
        card=card,
        settings=context.settings,
        batch_size=batch_size,
        labels=tuple(entry.label for entry in context.registry.entries),
        network=network,
        scratch_dir=context.scratch_dir,
        log=context.log,
    )


def _refuse_a_spectrogram_shape_the_network_does_not_take(card: ModelCard, network) -> None:
    # The network's input is (batch, height, width, channels).
    takes = tuple(network.input_shape[1:3])
    states = None if card.spectrogram_shape is None else tuple(card.spectrogram_shape)
    if states != takes:
        raise ValueError(
            f"card field spectrogram_shape is {states!r}, but the loaded network takes {takes!r}"
        )


class OwlModel:
    """OWL and its loaded network, run over one recording at a time."""

    def __init__(
        self,
        *,
        card: ModelCard,
        settings: Mapping[str, JsonScalar],
        batch_size: int,
        labels: tuple[str, ...],
        network: object,
        scratch_dir: Path,
        log: Log,
    ) -> None:
        self._card = card
        self._geometry = recipe(card, settings).audio.geometry
        self._batch_size = batch_size
        self._labels = labels
        self._network = network
        self._scratch_dir = scratch_dir
        self._log = log
        self._png_dir: Path | None = None

    def run(self, input: Input) -> Iterator[WindowOutput]:
        """Render every window's spectrogram, then yield each window's scores in order."""
        if not isinstance(input, AudioClip):
            raise TypeError(f"{OWNER} runs on audio, got a {type(input).__name__}")
        return self._run_windows(input.path)

    def _run_windows(self, path: Path) -> Iterator[WindowOutput]:
        bounds = window_bounds(sf.info(str(path)).duration, self._geometry)
        # An empty path list would reach TensorFlow as a float32 dataset, which it cannot read.
        if not bounds:
            return
        paths = self._render_spectrograms(path, bounds)
        for index, scores in _predict(self._network, paths, self._batch_size):
            start, end = bounds[index]
            # Every label in output order, unfiltered: the engine keeps what was asked
            # for. strict refuses a network that gives the registry more or fewer scores.
            scored = tuple(
                ClassScore(label=label, score=score)
                for label, score in zip(self._labels, scores, strict=True)
            )
            yield WindowOutput(start=start, end=end, scores=scored)

    def _render_spectrograms(
        self, path: Path, bounds: Sequence[tuple[float, float]]
    ) -> list[str]:
        """Render each window's spectrogram as a PNG file, and return the paths in order."""
        # Set before rendering, so after_recording removes a partly rendered recording.
        self._png_dir = Path(tempfile.mkdtemp(prefix="owl-spectrograms-", dir=self._scratch_dir))
        self._log(f"rendering {len(bounds)} spectrograms for {path.name}")
        pngs = []
        for index, (start, _) in enumerate(bounds):
            dest = self._png_dir / f"{index}.png"
            # A render failure raises: skipping the window would leave a hole in the recording.
            spectrogram_from_flac(
                str(path),
                start_time=start,
                duration=self._geometry.window_duration,
                shape=tuple(self._card.spectrogram_shape),
                dest=str(dest),
            )
            pngs.append(str(dest))
        return pngs

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
        predictions = _score_batch(network, images)
        for index, vector in zip(indices.numpy(), predictions):
            yield int(index), vector.tolist()


def _score_batch(network, images):
    """The network's scores for one batch, computed on the CPU."""
    with tf.device(TF_DEVICE):
        return network(images).numpy()
