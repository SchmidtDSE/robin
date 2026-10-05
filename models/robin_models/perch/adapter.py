"""The Perch v8 adapter. Importing this module loads TensorFlow."""

import math
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from fractions import Fraction
from pathlib import Path

from robin_contracts.cards import AudioGeometry, ModelCard, RunnerResampled
from robin_contracts.inputs import AudioClip, Input
from robin_contracts.protocols import JsonScalar, Log, ModelContext
from robin_contracts.records import ClassScore, WindowOutput
from robin_contracts.registry import TaxonRegistry
from robin_contracts.specs import recipe, window_bounds

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

DEFAULT_BATCH_SIZE = 32
DEVICE = "cpu"
TF_DEVICE = "/CPU:0"
SIGNATURE = "serving_default"
LABELS_HEADER = "ebird2021"

# Where each pinned file goes in the folder TensorFlow loads.
SAVED_MODEL_LAYOUT = {
    "saved_model": Path("saved_model.pb"),
    "variables_data": Path("variables") / "variables.data-00000-of-00001",
    "variables_index": Path("variables") / "variables.index",
}

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
    "score_domain": "probability",
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
    _refuse_a_card_perch_does_not_follow(card)
    batch_size = _read_batch_size(context.resources)
    _read_device(context.resources)
    _require_roles(context.files, needs_labels=context.registry is not None)
    folder = _saved_model_folder(context.files, context.scratch_dir)
    with tf.device(TF_DEVICE):
        loaded = tf.saved_model.load(str(folder))
    serving = _serving_signature(loaded)
    _require_input(serving, round(card.window_duration * card.sample_rate))
    _require_output(serving, "embedding", card.embedding_dim)
    labels = None
    if context.registry is not None:
        _require_output(serving, "label", len(context.registry.entries))
        labels = _labels_matching_the_registry(context.files["labels"], context.registry)
    return PerchModel(
        card=card,
        batch_size=batch_size,
        labels=labels,
        loaded=loaded,
        serving=serving,
        emit_embeddings=context.emit_embeddings,
        log=context.log,
    )


def _refuse_a_card_perch_does_not_follow(card: ModelCard) -> None:
    for field, does in BEHAVIOUR.items():
        states = getattr(card, field)
        if states != does:
            raise ValueError(f"card field {field} is {states!r}, but model perch/v8 does {does!r}")


def _read_batch_size(resources: Mapping[str, JsonScalar]) -> int:
    """The `batch_size` resource, or the default."""
    size = resources.get("batch_size", DEFAULT_BATCH_SIZE)
    # bool is an int subclass, and True is a type error, not a batch of one.
    if type(size) is not int or size < 1:
        raise ValueError(f"resource batch_size must be a positive int, got {size!r}")
    return size


def _read_device(resources: Mapping[str, JsonScalar]) -> None:
    device = resources.get("device", DEVICE)
    if device != DEVICE:
        raise ValueError(f"model perch/v8 runs only on resource device {DEVICE!r}, got {device!r}")


def _require_roles(files: Mapping[str, Path], *, needs_labels: bool) -> None:
    roles = [*SAVED_MODEL_LAYOUT, "labels"] if needs_labels else list(SAVED_MODEL_LAYOUT)
    for role in roles:
        if role not in files:
            raise ValueError(
                f"model perch/v8 needs a file under the role {role!r}, "
                f"and got the roles {sorted(files)}"
            )


def _saved_model_folder(files: Mapping[str, Path], scratch_dir: Path) -> Path:
    """A new folder laid out as TensorFlow expects, linking to the pinned files."""
    folder = Path(tempfile.mkdtemp(prefix="perch-savedmodel-", dir=scratch_dir))
    for role, place in SAVED_MODEL_LAYOUT.items():
        link = folder / place
        link.parent.mkdir(exist_ok=True)
        link.symlink_to(files[role])
    return folder


def _serving_signature(loaded) -> object:
    signatures = loaded.signatures
    if SIGNATURE not in signatures:
        raise ValueError(
            f"model perch/v8 needs the SavedModel signature {SIGNATURE!r}, "
            f"and found {sorted(signatures)}"
        )
    return signatures[SIGNATURE]


def _require_input(signature, width: int) -> None:
    inputs = signature.structured_input_signature[1]
    if list(inputs) != ["inputs"]:
        raise ValueError(
            f"model perch/v8 needs the {SIGNATURE} signature to take one input named 'inputs', "
            f"and it takes {sorted(inputs)}"
        )
    shape = inputs["inputs"].shape.as_list()
    if shape != [None, width]:
        raise ValueError(
            f"model perch/v8 needs the {SIGNATURE} signature's input shape to be "
            f"[None, {width}], and it is {shape}"
        )


def _require_output(signature, output: str, width: int) -> None:
    outputs = signature.structured_outputs
    shape = outputs[output].shape.as_list() if output in outputs else None
    if shape != [None, width]:
        raise ValueError(
            f"model perch/v8 needs the {SIGNATURE} signature's output {output!r} to have "
            f"shape [None, {width}], and found {shape}"
        )


def _labels_matching_the_registry(path: Path, registry: TaxonRegistry) -> tuple[str, ...]:
    """The registry's labels, once the labels file shipped with the weights equals them.

    The file is a header line, then one eBird code per line; Google ships it with CRLF endings.
    """
    try:
        lines = path.read_bytes().decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"model perch/v8's labels file {path} is not UTF-8") from error
    first = lines[0] if lines else None
    if first != LABELS_HEADER:
        raise ValueError(
            f"model perch/v8's labels file must start with the line {LABELS_HEADER!r}, "
            f"and its first line is {first!r}"
        )
    codes = lines[1:]
    labels = tuple(entry.label for entry in registry.entries)
    if len(codes) != len(labels):
        raise ValueError(
            f"model perch/v8's labels file has {len(codes)} labels, "
            f"but the registry has {len(labels)}"
        )
    for position, (code, label) in enumerate(zip(codes, labels)):
        if code != label:
            raise ValueError(
                f"model perch/v8's labels file has {code!r} at position {position}, "
                f"but the registry has {label!r}"
            )
    return labels


class PerchModel:
    """Perch and its loaded SavedModel, run over one recording at a time."""

    def __init__(
        self,
        *,
        card: ModelCard,
        batch_size: int,
        labels: tuple[str, ...] | None,
        loaded: object,
        serving: object,
        emit_embeddings: bool,
        log: Log,
    ) -> None:
        self._geometry = recipe(card).audio.geometry
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
        """Read, batch and score the recording one window at a time, yielding each in order."""
        if not isinstance(input, AudioClip):
            raise TypeError(f"model perch/v8 runs on audio, got a {type(input).__name__}")
        return self._windows(input.path)

    def _windows(self, path: Path) -> Iterator[WindowOutput]:
        with sf.SoundFile(str(path)) as audio:
            if audio.frames == 0:
                raise ValueError(f"model perch/v8 cannot run {path}: it has no audio frames")
            bounds = window_bounds(audio.frames / audio.samplerate, self._geometry)
            self._log(f"running {len(bounds)} windows of {path.name}")
            for first in range(0, len(bounds), self._batch_size):
                batch_bounds = bounds[first : first + self._batch_size]
                batch = self._batch(audio, batch_bounds)
                yield from self._window_outputs(batch_bounds, *self._infer(batch))

    def _batch(self, audio: "sf.SoundFile", bounds: Sequence[tuple[float, float]]) -> "np.ndarray":
        """The windows as rows, each mixed down, resampled and zero-padded at its end."""
        batch = np.zeros((len(bounds), self._width), dtype=np.float32)
        for row, (start, end) in zip(batch, bounds):
            window = _read_window_at_rate(audio, start, end, self._rate)
            row[: len(window)] = window
        return batch

    def _infer(self, batch: "np.ndarray") -> tuple["np.ndarray | None", "np.ndarray | None"]:
        """The batch's logits when there are labels, and its embeddings when asked for."""
        with tf.device(TF_DEVICE):
            tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
            outputs = self._serving(inputs=tensor)
        # Read by name: the outputs' order differs between TensorFlow versions.
        logits = outputs["label"].numpy() if self._labels is not None else None
        embeddings = outputs["embedding"].numpy() if self._emit_embeddings else None
        return logits, embeddings

    def _window_outputs(
        self,
        bounds: Sequence[tuple[float, float]],
        logits: "np.ndarray | None",
        embeddings: "np.ndarray | None",
    ) -> Iterator[WindowOutput]:
        if logits is None:
            score_rows = [()] * len(bounds)
        else:
            _refuse_unusable_logits(logits, bounds, self._labels)
            score_rows = [self._scored(row) for row in sigmoid(logits).tolist()]
        embedding_rows = [None] * len(bounds) if embeddings is None else list(embeddings)
        for (start, end), scores, embedding in zip(
            bounds, score_rows, embedding_rows, strict=True
        ):
            # Each window gets its own copy, so no two share the batch's memory.
            embedding = None if embedding is None else embedding.copy()
            yield WindowOutput(start=start, end=end, scores=scores, embedding=embedding)

    def _scored(self, scores: list[float]) -> tuple[ClassScore, ...]:
        return tuple(
            ClassScore(label=label, score=score) for label, score in zip(self._labels, scores)
        )

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
    first = _first_frame_on_the_shared_grid(start, rate, g)
    last = min(round(end * rate), frames)
    if rate == target_rate:
        return _mixed_down(_read_frames(audio, first, last))
    context_first = max(first - rate, 0)
    context_last = min(last + rate, frames)
    window = _mixed_down(_read_frames(audio, context_first, context_last))
    resampled = resample_poly(window, up, down)
    offset = context_first * up // down
    # The grid check makes first and context_first multiples of down, so these are exact;
    # -(-a // b) rounds up.
    keep_from = first * up // down - offset
    keep_to = -(-last * up // down) - offset
    return resampled[keep_from:keep_to].astype(np.float32)


def _first_frame_on_the_shared_grid(start: float, rate: int, g: int) -> int:
    """The window's first frame, once its start is a whole 32 kHz sample as well."""
    # Checked exactly: a start that is on the grid in decimal but not in binary is refused too.
    if (Fraction(start) * g).denominator != 1:
        raise ValueError(
            f"model perch/v8 cannot place the window at {start!r} s exactly: at {rate} Hz, "
            f"a start must be a whole number of 1/{g} s steps"
        )
    return int(Fraction(start) * rate)


def _read_frames(audio: "sf.SoundFile", first: int, last: int) -> "np.ndarray":
    """Frames `first` to `last`, as float32 with one column per channel."""
    audio.seek(first)
    return audio.read(last - first, dtype="float32", always_2d=True)


def _mixed_down(frames: "np.ndarray") -> "np.ndarray":
    """One channel as it is, or more averaged in float32."""
    if frames.shape[1] == 1:
        return frames[:, 0]
    return np.mean(frames, axis=1, dtype=np.float32)


def _refuse_unusable_logits(
    logits: "np.ndarray", bounds: Sequence[tuple[float, float]], labels: Sequence[str]
) -> None:
    """Refuse logits not shaped (windows, labels), or not finite."""
    if logits.shape != (len(bounds), len(labels)):
        raise ValueError(
            f"model perch/v8 gave logits of shape {logits.shape} for {len(bounds)} "
            f"windows of {len(labels)} labels"
        )
    found = np.argwhere(~np.isfinite(logits))
    if len(found):
        window, label = found[0]
        raise ValueError(
            f"model perch/v8 gave the logit {float(logits[window, label])!r} for "
            f"{labels[label]!r} in the window at {bounds[window][0]} s"
        )


def sigmoid(logits: "np.ndarray") -> "np.ndarray":
    """Scores from logits, computed in float32 without clipping.

    exp is only ever taken of a value at or below 0, so no logit overflows. Below about -103 a
    score underflows to exactly 0, which is expected and not reported.
    """
    x = np.asarray(logits, dtype=np.float32)
    with np.errstate(under="ignore"):
        e = np.exp(-np.abs(x), dtype=np.float32)
        return np.where(x >= 0, 1 / (1 + e), e / (1 + e))
