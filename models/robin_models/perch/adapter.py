"""The Perch v8 adapter. Importing this module loads TensorFlow."""

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction

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

OWNER = "model perch/v8"
_RUNTIME_MODULES = ("numpy", "scipy", "soundfile", "tensorflow")

with explain_a_missing_runtime(OWNER, "perch", _RUNTIME_MODULES):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import tensorflow as tf

from robin_models._adapter_support.audio_windows import (
    drop_padded_rows,
    get_window_outputs,
    mix_down,
    refuse_unusable_logits,
    run_windows,
)
from robin_models._adapter_support.batch_pipeline import Bounds, WindowResources
from robin_models._adapter_support.saved_model import (
    create_saved_model_folder,
    get_signature,
    require_input,
    require_output,
)
from robin_models.registration import SAVED_MODEL_LAYOUT

SIGNATURE = "serving_default"

# What Perch does whatever its card says. The SavedModel takes 5 seconds of mono audio at
# 32 kHz, mixed down by channel mean, resampled with SciPy's polyphase resampler with context
# either side so it equals resampling the whole recording, and zero-padded at the end; it
# scores every class, and emits 1,280 float32 embeddings.
BEHAVIOUR = {
    "window_duration": 5.0,
    "sample_rate": 32000,
    "audio": AudioGeometry(
        downmix="mean",
        resampler=RunnerResampled(algorithm="scipy_polyphase_with_context"),
        pad="centre_crop_end_pad",
    ),
    "score_domain": "sigmoid",
    "min_detection_threshold": 0.0,
    "spectrogram_shape": None,
    "can_emit_embeddings": True,
    "embedding_dim": 1280,
    "embedding_dtype": "float32",
    "backend": "pb-fp32",
}


class _Resources(WindowResources):
    batch_size: PositiveInt = 32


def build(context: ModelContext) -> "PerchModel":
    """Check the card, resources and files, load the SavedModel, and return the model."""
    card = context.card
    refuse_a_card_the_model_does_not_follow(card, BEHAVIOUR, OWNER)
    resources = get_resources(_Resources, context.resources, OWNER)
    device = get_tf_device(resources.device, tf, OWNER)
    require_roles(context.files, SAVED_MODEL_LAYOUT, OWNER)
    folder = create_saved_model_folder(context.files, context.scratch_dir, "perch-savedmodel-")
    with tf.device(device):
        loaded = tf.saved_model.load(str(folder))
    serving = get_signature(loaded, SIGNATURE, OWNER)
    require_input(serving, SIGNATURE, round(card.window_duration * card.sample_rate), OWNER)
    require_output(serving, SIGNATURE, "embedding", card.embedding_dim, OWNER)
    labels = None
    if context.registry is not None:
        require_output(serving, SIGNATURE, "label", len(context.registry.entries), OWNER)
        labels = tuple(entry.label for entry in context.registry.entries)
    return PerchModel(
        card=card,
        settings=context.settings,
        resources=resources,
        labels=labels,
        loaded=loaded,
        device=device,
        serving=serving,
        emit_embeddings=context.emit_embeddings,
        log=context.log,
    )


class PerchModel:
    """Perch and its loaded SavedModel, run over one recording at a time."""

    def __init__(
        self,
        *,
        card: ModelCard,
        settings: Mapping[str, JsonScalar],
        resources: WindowResources,
        labels: tuple[str, ...] | None,
        loaded: object,
        device: str,
        serving: object,
        emit_embeddings: bool,
        log: Log,
    ) -> None:
        self._geometry = recipe(card, settings).audio.geometry
        self._rate = card.sample_rate
        self._width = round(card.window_duration * card.sample_rate)
        self._resources = resources
        self._labels = labels
        # Kept so the signature's variables stay loaded.
        self._loaded = loaded
        self._device = device
        self._serving = serving
        self._emit_embeddings = emit_embeddings
        self._log = log

    def run(self, input: Input) -> Iterator[WindowOutput]:
        """Read and score the recording a batch at a time, yielding each window in order."""
        if not isinstance(input, AudioClip):
            raise TypeError(f"{OWNER} runs on audio, got a {type(input).__name__}")
        return run_windows(
            input.path,
            geometry=self._geometry,
            width=self._width,
            resources=self._resources,
            read_block=self._read_block,
            cut_window=_cut_window,
            run_batch=self._run_batch,
            log=self._log,
            owner=OWNER,
        )

    def _read_block(self, audio: "sf.SoundFile", bounds: Bounds) -> "_Block":
        """Read a block at this model's rate, which the window loop does not pass."""
        return _read_block_at_rate(audio, bounds, self._rate)

    def _infer(self, batch: "np.ndarray") -> tuple["np.ndarray | None", "np.ndarray | None"]:
        """The batch's logits when there are labels, and its embeddings when asked for."""
        with tf.device(self._device):
            tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
            outputs = self._serving(inputs=tensor)
        # Read by name: the outputs' order differs between TensorFlow versions.
        logits = outputs["label"].numpy() if self._labels is not None else None
        embeddings = outputs["embedding"].numpy() if self._emit_embeddings else None
        return logits, embeddings

    def _run_batch(self, bounds: Bounds, batch: "np.ndarray") -> Iterator[WindowOutput]:
        """Score one batch of padded rows, and make one output for each window."""
        logits, embeddings = self._infer(batch)
        logits = drop_padded_rows(logits, batch, bounds, "logits", OWNER)
        embeddings = drop_padded_rows(embeddings, batch, bounds, "embeddings", OWNER)
        scores = None
        if logits is not None:
            refuse_unusable_logits(logits, bounds, self._labels, OWNER)
            scores = get_sigmoid(logits)
        return get_window_outputs(bounds, scores, embeddings)

    def after_recording(self) -> None:
        """Nothing to do: a recording leaves no files behind, and the model stays loaded."""

    def clean_up(self) -> None:
        """Release the loaded SavedModel and its signature."""
        self._loaded = self._serving = None


@dataclass(frozen=True)
class _Block:
    """A block of windows at the model's rate.

    `samples[n]` is sample `offset + n` of the whole recording at the model's rate.
    `file_rate` is the file's sample rate, and `file_frames` is the number of frames in the
    whole file.
    `gcd` is the greatest common divisor of the two rates, and `up / down` is their ratio.
    """

    samples: "np.ndarray"
    offset: int
    file_rate: int
    file_frames: int
    gcd: int
    up: int
    down: int


def _read_block_at_rate(audio: "sf.SoundFile", bounds: Bounds, target_rate: int) -> _Block:
    """Decode the frames of every window in `bounds` at once, mix them down, and return
    them as a block at the model's rate.

    Away from 32 kHz, the block is read with up to a second of frames either side and
    resampled at once, with them, to float32. The polyphase filter reaches far less than a
    second, so each sample equals the same sample of the whole recording resampled at
    once, which is how Perch's training data was resampled.
    """
    rate, file_frames = audio.samplerate, audio.frames
    gcd = math.gcd(rate, target_rate)
    up, down = target_rate // gcd, rate // gcd
    first = _get_first_frame_on_grid(bounds[0][0], rate, gcd)
    last = max(min(round(end * rate), file_frames) for _, end in bounds)
    if rate == target_rate:
        samples, offset = mix_down(_read_frames(audio, first, last)), first
    else:
        context_first = max(first - rate, 0)
        context_last = min(last + rate, file_frames)
        decoded = mix_down(_read_frames(audio, context_first, context_last))
        samples = resample_poly(decoded, up, down).astype(np.float32)
        # The grid check makes context_first a multiple of down, so this is exact.
        offset = context_first * up // down
    return _Block(
        samples=samples,
        offset=offset,
        file_rate=rate,
        file_frames=file_frames,
        gcd=gcd,
        up=up,
        down=down,
    )


def _cut_window(block: _Block, start: float, end: float) -> "np.ndarray":
    """Return the window's samples at the model's rate, cut from its block and not yet padded."""
    first = _get_first_frame_on_grid(start, block.file_rate, block.gcd)
    last = min(round(end * block.file_rate), block.file_frames)
    # The grid check makes first a multiple of down, so this is exact; -(-a // b) rounds up.
    # At 32 kHz, up and down are 1, so this keeps the window's own frames.
    keep_from = first * block.up // block.down - block.offset
    keep_to = -(-last * block.up // block.down) - block.offset
    return block.samples[keep_from:keep_to]


def _get_first_frame_on_grid(start: float, rate: int, gcd: int) -> int:
    """The window's first frame, once its start is a whole 32 kHz sample as well."""
    # Checked exactly: a start that is on the grid in decimal but not in binary is refused too.
    if (Fraction(start) * gcd).denominator != 1:
        raise ValueError(
            f"{OWNER} cannot place the window at {start!r} s exactly: at {rate} Hz, "
            f"a start must be a whole number of 1/{gcd} s steps"
        )
    return int(Fraction(start) * rate)


def _read_frames(audio: "sf.SoundFile", first: int, last: int) -> "np.ndarray":
    """Frames `first` to `last`, as float32 with one column per channel."""
    audio.seek(first)
    return audio.read(last - first, dtype="float32", always_2d=True)


def get_sigmoid(logits: "np.ndarray") -> "np.ndarray":
    """Scores from logits, computed in float32 without clipping.

    exp is only ever taken of a value at or below 0, so no logit overflows. Below about -103 a
    score underflows to exactly 0, which is expected and not reported.
    """
    x = np.asarray(logits, dtype=np.float32)
    with np.errstate(under="ignore"):
        e = np.exp(-np.abs(x), dtype=np.float32)
        return np.where(x >= 0, 1 / (1 + e), e / (1 + e))
