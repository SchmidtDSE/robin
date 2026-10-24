"""The Perch v8 adapter. Importing this module loads TensorFlow."""

import math
from collections.abc import Iterator, Mapping
from fractions import Fraction

from robin_contracts.cards import AudioGeometry, ModelCard, RunnerResampled
from robin_contracts.inputs import AudioClip, Input
from robin_contracts.protocols import JsonScalar, Log, ModelContext
from robin_contracts.records import WindowOutput
from robin_contracts.specs import recipe
from robin_models._adapter_checks import (
    check_device,
    get_batch_size,
    refuse_a_card_the_model_does_not_follow,
)

_RUNTIME_MODULES = ("numpy", "scipy", "soundfile", "tensorflow")

try:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import tensorflow as tf
except ModuleNotFoundError as error:
    if (error.name or "").split(".")[0] not in _RUNTIME_MODULES:
        raise
    raise ModuleNotFoundError(
        f"model perch/v8 needs {error.name}, which is not installed; "
        f"install robin-models[perch]",
        name=error.name,
    ) from error

from robin_models._audio_windows import (
    Bounds,
    get_window_outputs,
    refuse_unusable_logits,
    run_windows,
)
from robin_models._saved_model import (
    create_saved_model_folder,
    get_signature,
    require_input,
    require_output,
    require_roles,
)

OWNER = "model perch/v8"
DEFAULT_BATCH_SIZE = 32
TF_DEVICE = "/CPU:0"
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


def build(context: ModelContext) -> "PerchModel":
    """Check the card, resources and files, load the SavedModel, and return the model."""
    card = context.card
    refuse_a_card_the_model_does_not_follow(card, BEHAVIOUR, OWNER)
    batch_size = get_batch_size(context.resources, DEFAULT_BATCH_SIZE)
    check_device(context.resources, OWNER)
    require_roles(context.files, OWNER)
    folder = create_saved_model_folder(context.files, context.scratch_dir, "perch-savedmodel-")
    with tf.device(TF_DEVICE):
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
        batch_size=batch_size,
        labels=labels,
        loaded=loaded,
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
        batch_size: int,
        labels: tuple[str, ...] | None,
        loaded: object,
        serving: object,
        emit_embeddings: bool,
        log: Log,
    ) -> None:
        self._geometry = recipe(card, settings).audio.geometry
        self._rate = card.sample_rate
        self._width = round(card.window_duration * card.sample_rate)
        self._batch_size = batch_size
        self._labels = labels
        # Kept so the signature's variables stay loaded.
        self._loaded = loaded
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
            batch_size=self._batch_size,
            read_window=self._read_window,
            run_batch=self._run_batch,
            log=self._log,
            owner=OWNER,
        )

    def _read_window(self, audio: "sf.SoundFile", start: float, end: float) -> "np.ndarray":
        return _read_window_at_rate(audio, start, end, self._rate)

    def _infer(self, batch: "np.ndarray") -> tuple["np.ndarray | None", "np.ndarray | None"]:
        """The batch's logits when there are labels, and its embeddings when asked for."""
        with tf.device(TF_DEVICE):
            tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
            outputs = self._serving(inputs=tensor)
        # Read by name: the outputs' order differs between TensorFlow versions.
        logits = outputs["label"].numpy() if self._labels is not None else None
        embeddings = outputs["embedding"].numpy() if self._emit_embeddings else None
        return logits, embeddings

    def _run_batch(self, bounds: Bounds, batch: "np.ndarray") -> Iterator[WindowOutput]:
        """Score one batch of padded rows, and make one output for each window."""
        logits, embeddings = self._infer(batch)
        scores = None
        if logits is not None:
            refuse_unusable_logits(logits, bounds, self._labels, OWNER)
            scores = get_sigmoid(logits)
        return get_window_outputs(bounds, scores, embeddings, self._labels)

    def after_recording(self) -> None:
        """Nothing to do: a recording leaves no files behind, and the model stays loaded."""

    def clean_up(self) -> None:
        """Release the loaded SavedModel and its signature."""
        self._loaded = self._serving = None


def _read_window_at_rate(
    audio: "sf.SoundFile", start: float, end: float, target_rate: int
) -> "np.ndarray":
    """The window's samples at the model's rate, mixed down and not yet padded.

    Away from 32 kHz, the window is read with up to a second of frames either side and
    resampled with them, then trimmed to its own samples. The polyphase filter reaches far less
    than a second, so the result equals the same samples of the whole recording resampled at
    once, which is how Perch's training data was resampled.
    """
    rate, frames = audio.samplerate, audio.frames
    g = math.gcd(rate, target_rate)
    up, down = target_rate // g, rate // g
    first = _get_first_frame_on_grid(start, rate, g)
    last = min(round(end * rate), frames)
    if rate == target_rate:
        return _mix_down(_read_frames(audio, first, last))
    context_first = max(first - rate, 0)
    context_last = min(last + rate, frames)
    window = _mix_down(_read_frames(audio, context_first, context_last))
    resampled = resample_poly(window, up, down)
    offset = context_first * up // down
    # The grid check makes first and context_first multiples of down, so these are exact;
    # -(-a // b) rounds up.
    keep_from = first * up // down - offset
    keep_to = -(-last * up // down) - offset
    return resampled[keep_from:keep_to].astype(np.float32)


def _get_first_frame_on_grid(start: float, rate: int, g: int) -> int:
    """The window's first frame, once its start is a whole 32 kHz sample as well."""
    # Checked exactly: a start that is on the grid in decimal but not in binary is refused too.
    if (Fraction(start) * g).denominator != 1:
        raise ValueError(
            f"{OWNER} cannot place the window at {start!r} s exactly: at {rate} Hz, "
            f"a start must be a whole number of 1/{g} s steps"
        )
    return int(Fraction(start) * rate)


def _read_frames(audio: "sf.SoundFile", first: int, last: int) -> "np.ndarray":
    """Frames `first` to `last`, as float32 with one column per channel."""
    audio.seek(first)
    return audio.read(last - first, dtype="float32", always_2d=True)


def _mix_down(frames: "np.ndarray") -> "np.ndarray":
    """One channel as it is, or more averaged in float32."""
    if frames.shape[1] == 1:
        return frames[:, 0]
    return np.mean(frames, axis=1, dtype=np.float32)


def get_sigmoid(logits: "np.ndarray") -> "np.ndarray":
    """Scores from logits, computed in float32 without clipping.

    exp is only ever taken of a value at or below 0, so no logit overflows. Below about -103 a
    score underflows to exactly 0, which is expected and not reported.
    """
    x = np.asarray(logits, dtype=np.float32)
    with np.errstate(under="ignore"):
        e = np.exp(-np.abs(x), dtype=np.float32)
        return np.where(x >= 0, 1 / (1 + e), e / (1 + e))
