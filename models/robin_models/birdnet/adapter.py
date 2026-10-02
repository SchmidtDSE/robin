"""The BirdNET v2.4 adapter. Importing this module loads TensorFlow."""

import tempfile
from collections.abc import Iterator, Mapping, Sequence
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
    from scipy.signal import resample
    import tensorflow as tf
except ModuleNotFoundError as error:
    if (error.name or "").split(".")[0] not in _RUNTIME_MODULES:
        raise
    raise ModuleNotFoundError(
        f"model birdnet/v2p4 needs {error.name}, which is not installed; "
        f"install robin-models[birdnet]",
        name=error.name,
    ) from error

DEFAULT_BATCH_SIZE = 32
DEVICE = "cpu"
TF_DEVICE = "/CPU:0"

# Where each pinned file goes in the folder TensorFlow loads.
SAVED_MODEL_LAYOUT = {
    "saved_model": Path("saved_model.pb"),
    "variables_data": Path("variables") / "variables.data-00000-of-00001",
    "variables_index": Path("variables") / "variables.index",
}

# What BirdNET does whatever its card says. The SavedModel takes 3 seconds of mono audio
# at 48 kHz, mixed down by channel mean, resampled per window with SciPy's FFT resampler
# and zero-padded at the end; it scores every class, and emits 1,024 float32 embeddings.
BEHAVIOUR = {
    "window_duration": 3.0,
    "sample_rate": 48000,
    "audio": AudioGeometry(
        downmix="mean",
        resampler=RunnerResampled(algorithm="scipy_fft_per_window"),
        pad="centre_crop_end_pad",
    ),
    "score_domain": "probability",
    "min_detection_threshold": 0.0,
    "spectrogram_shape": None,
    "can_emit_embeddings": True,
    "embedding_dim": 1024,
    "embedding_dtype": "float32",
    "backend": "pb-fp32",
}


def build(context: ModelContext) -> "BirdnetModel":
    """Check the card, resources and files, load the SavedModel, and return the model."""
    card = context.card
    _refuse_a_card_birdnet_does_not_follow(card)
    batch_size = _read_batch_size(context.resources)
    _read_device(context.resources)
    _require_roles(context.files, needs_labels=context.registry is not None)
    folder = _saved_model_folder(context.files, context.scratch_dir)
    with tf.device(TF_DEVICE):
        loaded = tf.saved_model.load(str(folder))
    basic, embeddings = _signatures(loaded)
    window = round(card.window_duration * card.sample_rate)
    _require_input(basic, "basic", window)
    _require_input(embeddings, "embeddings", window)
    _require_output(embeddings, "embeddings", "embeddings", card.embedding_dim)
    labels = None
    if context.registry is not None:
        _require_output(basic, "basic", "scores", len(context.registry.entries))
        labels = _labels_matching_the_registry(context.files["labels"], context.registry)
    return BirdnetModel(
        card=card,
        batch_size=batch_size,
        labels=labels,
        loaded=loaded,
        basic=basic,
        embeddings=embeddings,
        emit_embeddings=context.emit_embeddings,
        log=context.log,
    )


def _refuse_a_card_birdnet_does_not_follow(card: ModelCard) -> None:
    for field, does in BEHAVIOUR.items():
        states = getattr(card, field)
        if states != does:
            raise ValueError(
                f"card field {field} is {states!r}, but model birdnet/v2p4 does {does!r}"
            )


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
        raise ValueError(
            f"model birdnet/v2p4 runs only on resource device {DEVICE!r}, got {device!r}"
        )


def _require_roles(files: Mapping[str, Path], *, needs_labels: bool) -> None:
    roles = [*SAVED_MODEL_LAYOUT, "labels"] if needs_labels else list(SAVED_MODEL_LAYOUT)
    for role in roles:
        if role not in files:
            raise ValueError(
                f"model birdnet/v2p4 needs a file under the role {role!r}, "
                f"and got the roles {sorted(files)}"
            )


def _saved_model_folder(files: Mapping[str, Path], scratch_dir: Path) -> Path:
    """A new folder laid out as TensorFlow expects, linking to the pinned files."""
    folder = Path(tempfile.mkdtemp(prefix="birdnet-savedmodel-", dir=scratch_dir))
    for role, place in SAVED_MODEL_LAYOUT.items():
        link = folder / place
        link.parent.mkdir(exist_ok=True)
        link.symlink_to(files[role])
    return folder


def _signatures(loaded) -> tuple[object, object]:
    signatures = loaded.signatures
    found = sorted(signatures)
    if "basic" not in signatures or "embeddings" not in signatures:
        raise ValueError(
            "model birdnet/v2p4 needs the SavedModel signatures 'basic' and 'embeddings', "
            f"and found {found}"
        )
    return signatures["basic"], signatures["embeddings"]


def _require_input(signature, name: str, width: int) -> None:
    inputs = signature.structured_input_signature[1]
    if list(inputs) != ["inputs"]:
        raise ValueError(
            f"model birdnet/v2p4 needs the {name} signature to take one input named 'inputs', "
            f"and it takes {sorted(inputs)}"
        )
    shape = inputs["inputs"].shape.as_list()
    if shape != [None, width]:
        raise ValueError(
            f"model birdnet/v2p4 needs the {name} signature's input shape to be "
            f"[None, {width}], and it is {shape}"
        )


def _require_output(signature, name: str, output: str, width: int) -> None:
    outputs = signature.structured_outputs
    shape = outputs[output].shape.as_list() if output in outputs else None
    if shape != [None, width]:
        raise ValueError(
            f"model birdnet/v2p4 needs the {name} signature's output {output!r} to have "
            f"shape [None, {width}], and found {shape}"
        )


def _labels_matching_the_registry(path: Path, registry: TaxonRegistry) -> tuple[str, ...]:
    """The registry's labels, once the labels file shipped with the weights equals them."""
    try:
        lines = path.read_bytes().decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"model birdnet/v2p4's labels file {path} is not UTF-8") from error
    labels = tuple(entry.label for entry in registry.entries)
    if len(lines) != len(labels):
        raise ValueError(
            f"model birdnet/v2p4's labels file has {len(lines)} labels, "
            f"but the registry has {len(labels)}"
        )
    for position, (line, label) in enumerate(zip(lines, labels)):
        if line != label:
            raise ValueError(
                f"model birdnet/v2p4's labels file has {line!r} at position {position}, "
                f"but the registry has {label!r}"
            )
    return labels


class BirdnetModel:
    """BirdNET and its loaded SavedModel, run over one recording at a time."""

    def __init__(
        self,
        *,
        card: ModelCard,
        batch_size: int,
        labels: tuple[str, ...] | None,
        loaded: object,
        basic: object,
        embeddings: object,
        emit_embeddings: bool,
        log: Log,
    ) -> None:
        self._geometry = recipe(card).audio.geometry
        self._rate = card.sample_rate
        self._width = round(card.window_duration * card.sample_rate)
        self._batch_size = batch_size
        self._labels = labels
        self._loaded = loaded
        self._basic = basic
        self._embeddings = embeddings
        self._emit_embeddings = emit_embeddings
        self._log = log

    def run(self, input: Input) -> Iterator[WindowOutput]:
        """Read, batch and score the recording one window at a time, yielding each in order."""
        if not isinstance(input, AudioClip):
            raise TypeError(f"model birdnet/v2p4 runs on audio, got a {type(input).__name__}")
        return self._windows(input.path)

    def _windows(self, path: Path) -> Iterator[WindowOutput]:
        info = sf.info(str(path))
        if info.frames == 0:
            raise ValueError(f"model birdnet/v2p4 cannot run {path}: it has no audio frames")
        bounds = window_bounds(info.frames / info.samplerate, self._geometry)
        self._log(f"running {len(bounds)} windows of {path.name}")
        with sf.SoundFile(str(path)) as audio:
            for first in range(0, len(bounds), self._batch_size):
                batch_bounds = bounds[first : first + self._batch_size]
                batch = self._batch(audio, batch_bounds)
                yield from self._window_outputs(batch_bounds, *self._infer(batch))

    def _batch(self, audio: "sf.SoundFile", bounds: Sequence[tuple[float, float]]) -> "np.ndarray":
        """The windows as rows, each mixed down, resampled and zero-padded at its end."""
        batch = np.zeros((len(bounds), self._width), dtype=np.float32)
        for row, (start, end) in zip(batch, bounds):
            window = _mixed_down(_read_window(audio, start, end))
            window = _resampled(window, audio.samplerate, self._rate, start)
            if len(window) > self._width:
                raise ValueError(
                    f"model birdnet/v2p4 read {len(window)} samples for the window at {start} s, "
                    f"more than the {self._width} it takes"
                )
            row[: len(window)] = window
        return batch

    def _infer(self, batch: "np.ndarray") -> tuple["np.ndarray | None", "np.ndarray | None"]:
        """The batch's logits when there are labels, and its embeddings when asked for."""
        logits = embeddings = None
        with tf.device(TF_DEVICE):
            tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
            if self._labels is not None:
                logits = self._basic(inputs=tensor)["scores"].numpy()
            if self._emit_embeddings:
                embeddings = self._embeddings(inputs=tensor)["embeddings"].numpy()
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
            _refuse_a_non_finite_logit(logits, bounds, self._labels)
            score_rows = [self._scored(row) for row in sigmoid(logits).tolist()]
        embedding_rows = [None] * len(bounds) if embeddings is None else list(embeddings)
        for (start, end), scores, embedding in zip(
            bounds, score_rows, embedding_rows, strict=True
        ):
            # Each window gets its own copy, so no two share the batch's memory.
            embedding = None if embedding is None else embedding.copy()
            yield WindowOutput(start=start, end=end, scores=scores, embedding=embedding)

    def _scored(self, scores: list[float]) -> tuple[ClassScore, ...]:
        # strict refuses a model that gives the registry more or fewer scores.
        return tuple(
            ClassScore(label=label, score=score)
            for label, score in zip(self._labels, scores, strict=True)
        )

    def after_recording(self) -> None:
        """Nothing to do: a recording leaves no files behind, and the model stays loaded."""

    def clean_up(self) -> None:
        """Release the loaded SavedModel and its signatures."""
        self._loaded = self._basic = self._embeddings = None


def _read_window(audio: "sf.SoundFile", start: float, end: float) -> "np.ndarray":
    """The window's frames, read on their own, as float32 with one column per channel."""
    rate = audio.samplerate
    first = round(start * rate)
    audio.seek(first)
    return audio.read(
        min(round(end * rate), audio.frames) - first, dtype="float32", always_2d=True
    )


def _mixed_down(frames: "np.ndarray") -> "np.ndarray":
    """One channel as it is, or more averaged in float32, as birdnet 0.2.12's convert_to_mono."""
    if frames.shape[1] == 1:
        return frames[:, 0]
    return np.mean(frames, axis=1, dtype=np.float32)


def _resampled(window: "np.ndarray", rate: int, target_rate: int, start: float) -> "np.ndarray":
    """The window at the model's rate, as birdnet 0.2.12's resample_array_by_sr."""
    if rate == target_rate:
        return window
    target = round(len(window) / rate * target_rate)
    if target == 0:
        raise ValueError(
            f"model birdnet/v2p4 cannot resample the window at {start} s: its {len(window)} "
            f"frames at {rate} Hz resample to {target} samples at {target_rate} Hz"
        )
    return resample(window, target)


def _refuse_a_non_finite_logit(
    logits: "np.ndarray", bounds: Sequence[tuple[float, float]], labels: Sequence[str]
) -> None:
    # Checked before the sigmoid's clip, which would turn +inf into a confident score.
    if logits.shape != (len(bounds), len(labels)):
        raise ValueError(
            f"model birdnet/v2p4 gave logits of shape {logits.shape} for {len(bounds)} "
            f"windows of {len(labels)} labels"
        )
    found = np.argwhere(~np.isfinite(logits))
    if len(found):
        window, label = found[0]
        raise ValueError(
            f"model birdnet/v2p4 gave the logit {float(logits[window, label])!r} for "
            f"{labels[label]!r} in the window at {bounds[window][0]} s"
        )


def sigmoid(logits: "np.ndarray") -> "np.ndarray":
    """Scores from logits, computed in float32 as birdnet 0.2.12's flat_sigmoid_logaddexp_fast
    is with its default sensitivity: logits are clipped to -15..15 first."""
    y = -np.clip(np.asarray(logits, dtype=np.float32), -15.0, 15.0)
    e = np.exp(-np.abs(y), dtype=np.float32)
    return np.where(y >= 0, e / (1 + e), 1 / (1 + e))
