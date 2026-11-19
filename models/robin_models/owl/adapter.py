"""The OWL (PNW-Cnet v4) adapter. Importing this module loads TensorFlow."""

import math
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from pydantic import PositiveInt
from robin_contracts.cards import AudioGeometry, ModelCard, RunnerResampled
from robin_contracts.inputs import AudioClip, Input
from robin_contracts.protocols import JsonScalar, Log, ModelContext
from robin_contracts.records import WindowOutput
from robin_contracts.specs import recipe
from robin_models._adapter_support.adapter_setup import (
    explain_a_missing_runtime,
    get_resources,
    get_tf_device,
    refuse_a_card_the_model_does_not_follow,
    require_roles,
)
from robin_models.registration import WEIGHTS_ROLE

# Must be set before TensorFlow is imported: the weights are a Keras 2 file.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
# soxr resamples with OpenMP threads, which use much more CPU than they save. The reader
# threads are limited by the CPU, so soxr gets one thread unless the caller set a number.
# OpenMP reads this value when it loads, so it must be set before sox_numpy loads soxr.
os.environ.setdefault("OMP_NUM_THREADS", "1")

OWNER = "model owl/v4"
_RUNTIME_MODULES = ("numpy", "soundfile", "tensorflow", "tf_keras", "sox_numpy")

with explain_a_missing_runtime(OWNER, "owl", _RUNTIME_MODULES):
    import numpy as np
    import soundfile as sf
    import tensorflow as tf
    import tf_keras  # noqa: F401  Legacy Keras runs from it.
    # spectrogram() returns palette indices, and get_palette() returns the palette that
    # gives each index its gray. See _get_grays.
    from sox_numpy import get_palette, spectrogram

from robin_models._adapter_support.audio_windows import get_window_outputs, run_windows
from robin_models._adapter_support.batch_pipeline import Bounds, WindowResources

# What OWL does whatever its card says. sox_numpy renders at 8 kHz from the first
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


class _Resources(WindowResources):
    # Batch size changes the scores at float32 rounding level, and nothing else.
    batch_size: PositiveInt = 64


def build(context: ModelContext) -> "OwlModel":
    """Check the card, resources and files, load the weights, and return the model."""
    card = context.card
    refuse_a_card_the_model_does_not_follow(card, BEHAVIOUR, OWNER)
    resources = get_resources(_Resources, context.resources, OWNER)
    device = get_tf_device(resources.device, tf, OWNER)
    if context.registry is None:
        raise ValueError(f"{OWNER} needs a registry to name its outputs, and got none")
    require_roles(context.files, [WEIGHTS_ROLE], OWNER)
    grays = _get_grays(get_palette())
    with tf.device(device):
        network = tf.keras.models.load_model(str(context.files[WEIGHTS_ROLE]))
    _refuse_a_spectrogram_shape_the_network_does_not_take(card, network)
    return OwlModel(
        card=card,
        settings=context.settings,
        resources=resources,
        labels=tuple(entry.label for entry in context.registry.entries),
        network=network,
        device=device,
        grays=grays,
        log=context.log,
    )


def _get_grays(palette: list[int]) -> "np.ndarray":
    """Return a table of the network's input value for each palette index.

    Each pixel that sox_numpy's `spectrogram()` returns is a palette index, not a
    brightness. Indices 0 to 3 are sox's chart colours, which a spectrogram does not use.
    Indices 4 to 254 are the loudness, from quiet to loud. The palette gives each index a
    colour, and every colour is a gray from 0 (black) to 255 (white).

    The network was trained on sox's PNG files. Each PNG stores the indices and this
    palette. The old adapter read each PNG back as gray values, and divided them by 255.
    So entry `i` of the table is the gray of index `i`, divided by 255, in float32.

    `palette` is flat RGB: the colour of index `i` is entries `3 * i` to `3 * i + 2`.
    Raises ValueError if the palette does not have 255 colours, or if a colour is not gray.
    """
    colours = np.asarray(palette, dtype=np.float32)
    if colours.shape != (255 * 3,):
        raise ValueError(
            f"{OWNER} needs a palette of 765 values (255 RGB colours), and got {len(palette)}"
        )
    colours = colours.reshape(-1, 3)
    if not np.all(colours == colours[:, :1]):
        raise ValueError(f"{OWNER} needs a palette of grays, and a colour is not gray")
    return colours[:, 0] / np.float32(255)


def _refuse_a_spectrogram_shape_the_network_does_not_take(card: ModelCard, network) -> None:
    # The network's input is (batch, height, width, channels).
    takes = tuple(network.input_shape[1:3])
    states = None if card.spectrogram_shape is None else tuple(card.spectrogram_shape)
    if states != takes:
        raise ValueError(
            f"card field spectrogram_shape is {states!r}, but the loaded network takes {takes!r}"
        )


@dataclass(frozen=True)
class _Block:
    """The first channel of a block of windows, from frame `first` of a file at `rate`."""

    first: int
    samples: "np.ndarray"
    rate: int


class OwlModel:
    """OWL and its loaded network, run over one recording at a time."""

    def __init__(
        self,
        *,
        card: ModelCard,
        settings: Mapping[str, JsonScalar],
        resources: WindowResources,
        labels: tuple[str, ...],
        network: object,
        device: str,
        grays: "np.ndarray",
        log: Log,
    ) -> None:
        self._geometry = recipe(card, settings).audio.geometry
        self._shape = tuple(card.spectrogram_shape)
        # Each row of a batch holds one window's spectrogram, flattened.
        self._row_width = math.prod(self._shape)
        self._resources = resources
        self._labels = labels
        self._network = network
        self._device = device
        self._grays = grays
        self._log = log

    def run(self, input: Input) -> Iterator[WindowOutput]:
        """Read, render and score the recording a batch at a time, yielding each window in
        order."""
        if not isinstance(input, AudioClip):
            raise TypeError(f"{OWNER} runs on audio, got a {type(input).__name__}")
        return run_windows(
            input.path,
            geometry=self._geometry,
            width=self._row_width,
            resources=self._resources,
            read_block=self._read_block,
            cut_window=self._cut_window,
            run_batch=self._run_batch,
            log=self._log,
            owner=OWNER,
        )

    def _get_frame_span(self, rate: int, start: float) -> tuple[int, int]:
        """Return the window's first frame and its number of frames, as sox_numpy's
        `load_audio` counts them."""
        return round(start * rate), round(self._geometry.window_duration * rate)

    def _read_block(self, audio: "sf.SoundFile", bounds: Bounds) -> _Block:
        """Decode the frames of every window in `bounds` at once, as float64, and keep the
        first channel."""
        rate = audio.samplerate
        first, _ = self._get_frame_span(rate, bounds[0][0])
        last_first, count = self._get_frame_span(rate, bounds[-1][0])
        audio.seek(first)
        decoded = audio.read(last_first + count - first, dtype="float64", always_2d=True)
        return _Block(first=first, samples=np.ascontiguousarray(decoded[:, 0]), rate=rate)

    def _cut_window(self, block: _Block, start: float, end: float) -> "np.ndarray":
        """Return the window's spectrogram as one row of the network's input values.

        The window is resampled on its own, as sox_numpy's `spectrogram_from_flac` does,
        so the block changes only how the audio is decoded. sox_numpy uses no TensorFlow,
        so the spectrogram is made on the CPU, in the reader thread.
        """
        window_first, count = self._get_frame_span(block.rate, start)
        at = window_first - block.first
        samples = block.samples[at : at + count]
        # A render failure raises: skipping the window would leave a hole in the recording.
        indices = spectrogram(samples, self._shape, sample_rate=block.rate)
        # Change each palette index to the value that the network expects.
        return self._grays[indices].reshape(-1)

    def _run_batch(self, bounds: Bounds, batch: "np.ndarray") -> Iterator[WindowOutput]:
        """Score the batch's windows, and make one output for each window.

        Only the windows' rows reach the network, not the zero rows after them: the rows
        in a batch change the scores at float32 rounding level.
        """
        height, width = self._shape
        images = batch[: len(bounds)].reshape(len(bounds), height, width, 1)
        with tf.device(self._device):
            scores = self._network(tf.convert_to_tensor(images, dtype=tf.float32)).numpy()
        # Every label in output order, unfiltered: the engine keeps what was asked for.
        if scores.shape != (len(bounds), len(self._labels)):
            raise ValueError(
                f"{OWNER} gave scores of shape {scores.shape} for {len(bounds)} windows, "
                f"but the registry has {len(self._labels)} labels"
            )
        return get_window_outputs(bounds, scores, None)

    def after_recording(self) -> None:
        """Nothing to do: a recording leaves no files behind, and the network stays loaded."""

    def clean_up(self) -> None:
        """Nothing to release: the network stays loaded until the process ends."""
