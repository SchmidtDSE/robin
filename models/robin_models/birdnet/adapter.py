"""The BirdNET v2.4 adapter. Importing this module loads TensorFlow."""

from collections.abc import Iterator, Mapping

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

OWNER = "model birdnet/v2p4"
DEFAULT_BATCH_SIZE = 32
TF_DEVICE = "/CPU:0"
SIGNATURE = "embeddings"

# The SavedModel's last layer, which turns an embedding into logits.
CLASSIFIER_KERNEL = "CLASS_DENSE_LAYER/kernel:0"
CLASSIFIER_BIAS = "CLASS_DENSE_LAYER/bias:0"

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
    "score_domain": "sigmoid",
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
    refuse_a_card_the_model_does_not_follow(card, BEHAVIOUR, OWNER)
    batch_size = get_batch_size(context.resources, DEFAULT_BATCH_SIZE)
    check_device(context.resources, OWNER)
    require_roles(context.files, OWNER)
    folder = create_saved_model_folder(context.files, context.scratch_dir, "birdnet-savedmodel-")
    with tf.device(TF_DEVICE):
        loaded = tf.saved_model.load(str(folder))
    embeddings = get_signature(loaded, SIGNATURE, OWNER)
    require_input(embeddings, SIGNATURE, round(card.window_duration * card.sample_rate), OWNER)
    require_output(embeddings, SIGNATURE, "embeddings", card.embedding_dim, OWNER)
    labels = kernel = bias = None
    if context.registry is not None:
        kernel, bias = _get_classifier_layer(
            loaded, card.embedding_dim, len(context.registry.entries)
        )
        labels = tuple(entry.label for entry in context.registry.entries)
    return BirdnetModel(
        card=card,
        settings=context.settings,
        batch_size=batch_size,
        labels=labels,
        loaded=loaded,
        embeddings=embeddings,
        kernel=kernel,
        bias=bias,
        emit_embeddings=context.emit_embeddings,
        log=context.log,
    )


def _get_classifier_layer(loaded, embedding_dim: int, labels: int) -> tuple[object, object]:
    """The kernel and bias of the layer that turns an embedding into one logit per label."""
    model = getattr(loaded, "model", None)
    if model is None:
        raise ValueError(
            f"{OWNER} needs the loaded SavedModel to have the attribute 'model', "
            "which holds its classifier layer"
        )
    variables = {variable.name: variable for variable in model.variables}
    kernel = _require_variable(variables, CLASSIFIER_KERNEL, [embedding_dim, labels])
    bias = _require_variable(variables, CLASSIFIER_BIAS, [labels])
    return kernel, bias


def _require_variable(variables: Mapping[str, object], name: str, shape: list[int]) -> object:
    if name not in variables:
        raise ValueError(
            f"{OWNER} needs the SavedModel variable {name!r}, and found none"
        )
    found = variables[name].shape.as_list()
    if found != shape:
        raise ValueError(
            f"{OWNER} needs the SavedModel variable {name!r} to have shape {shape}, "
            f"and it is {found}"
        )
    return variables[name]


class BirdnetModel:
    """BirdNET and its loaded SavedModel, run over one recording at a time."""

    def __init__(
        self,
        *,
        card: ModelCard,
        settings: Mapping[str, JsonScalar],
        batch_size: int,
        labels: tuple[str, ...] | None,
        loaded: object,
        embeddings: object,
        kernel: object | None,
        bias: object | None,
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
        self._embeddings = embeddings
        self._kernel = kernel
        self._bias = bias
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
        """The window mixed down and resampled, and not longer than the network takes."""
        window = _mix_down(_read_frames(audio, start, end))
        window = _resample_window(window, audio.samplerate, self._rate, start)
        if len(window) > self._width:
            raise ValueError(
                f"{OWNER} read {len(window)} samples for the window at {start} s, "
                f"more than the {self._width} it takes"
            )
        return window

    def _infer(self, batch: "np.ndarray") -> tuple["np.ndarray | None", "np.ndarray | None"]:
        """The batch's logits when there are labels, and its embeddings when asked for."""
        logits = None
        with tf.device(TF_DEVICE):
            tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
            embeddings = self._embeddings(inputs=tensor)["embeddings"]
            if self._labels is not None:
                # The scores are the model's classifier layer on the embedding, so one pass
                # gives both.
                logits = (tf.matmul(embeddings, self._kernel) + self._bias).numpy()
        embeddings = embeddings.numpy() if self._emit_embeddings else None
        return logits, embeddings

    def _run_batch(self, bounds: Bounds, batch: "np.ndarray") -> Iterator[WindowOutput]:
        """Score one batch of padded rows, and make one output for each window."""
        logits, embeddings = self._infer(batch)
        scores = None
        if logits is not None:
            # Checked before the sigmoid's clip, which would turn +inf into a confident score.
            refuse_unusable_logits(logits, bounds, self._labels, OWNER)
            scores = get_sigmoid(logits)
        return get_window_outputs(bounds, scores, embeddings, self._labels)

    def after_recording(self) -> None:
        """Nothing to do: a recording leaves no files behind, and the model stays loaded."""

    def clean_up(self) -> None:
        """Release the loaded SavedModel, its signature and its classifier layer."""
        self._loaded = self._embeddings = self._kernel = self._bias = None


def _read_frames(audio: "sf.SoundFile", start: float, end: float) -> "np.ndarray":
    """The window's frames, read on their own, as float32 with one column per channel."""
    rate = audio.samplerate
    first = round(start * rate)
    # Counted from the first frame: rounding both ends can read one frame too many or too few.
    audio.seek(first)
    return audio.read(
        min(round((end - start) * rate), audio.frames - first), dtype="float32", always_2d=True
    )


def _mix_down(frames: "np.ndarray") -> "np.ndarray":
    """One channel as it is, or more averaged in float32, as birdnet 0.2.12's convert_to_mono."""
    if frames.shape[1] == 1:
        return frames[:, 0]
    return np.mean(frames, axis=1, dtype=np.float32)


def _resample_window(
    window: "np.ndarray", rate: int, target_rate: int, start: float
) -> "np.ndarray":
    """The window at the model's rate, as birdnet 0.2.12's resample_array_by_sr."""
    if rate == target_rate:
        return window
    target = round(len(window) / rate * target_rate)
    if target == 0:
        raise ValueError(
            f"{OWNER} cannot resample the window at {start} s: its {len(window)} "
            f"frames at {rate} Hz resample to {target} samples at {target_rate} Hz"
        )
    return resample(window, target)


def get_sigmoid(logits: "np.ndarray") -> "np.ndarray":
    """Scores from logits, computed in float32 as birdnet 0.2.12's flat_sigmoid_logaddexp_fast
    is with its default sensitivity: logits are clipped to -15..15 first."""
    y = -np.clip(np.asarray(logits, dtype=np.float32), -15.0, 15.0)
    e = np.exp(-np.abs(y), dtype=np.float32)
    return np.where(y >= 0, e / (1 + e), 1 / (1 + e))
