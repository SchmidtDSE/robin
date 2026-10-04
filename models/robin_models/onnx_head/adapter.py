"""The ONNX head runtime. Importing this module loads onnxruntime."""

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from robin_contracts.cards import HeadCard, ModelCard
from robin_contracts.inputs import Embeddings, Input
from robin_contracts.protocols import JsonScalar, Log, ModelContext
from robin_contracts.records import ClassScore, WindowOutput
from robin_contracts.registry import TaxonRegistry
from robin_contracts.work import REGISTRY_ROLE

_RUNTIME_MODULES = ("numpy", "onnxruntime")

try:
    import numpy as np
    import onnxruntime
except ModuleNotFoundError as error:
    if (error.name or "").split(".")[0] not in _RUNTIME_MODULES:
        raise
    raise ModuleNotFoundError(
        f"head runtime onnx needs {error.name}, which is not installed; "
        f"install robin-models[onnx-head]",
        name=error.name,
    ) from error

RUNTIME = "onnx"
GRAPH_ROLE = "graph"
DEFAULT_BATCH_SIZE = 256
DEVICE = "cpu"
PROVIDERS = ["CPUExecutionProvider"]
FLOAT32 = "tensor(float)"


def build(context: ModelContext) -> "OnnxHead":
    """Check the card, resources and files, load the graph, check it, and return the head."""
    card = context.card
    _refuse_a_card_this_runtime_does_not_run(card)
    batch_size = _read_batch_size(context.resources)
    _read_device(context.resources)
    _require_roles(context.files, context.registry)
    session = onnxruntime.InferenceSession(str(context.files[GRAPH_ROLE]), providers=PROVIDERS)
    labels = tuple(entry.label for entry in context.registry.entries)
    dim, count = card.embedding_dim, len(labels)
    input_name = _only_node(
        session.get_inputs(), "input", dim, f"the card's embedding_dim {dim}"
    )
    output_name = _only_node(
        session.get_outputs(), "output", count, f"the registry's {count} labels"
    )
    return OnnxHead(
        session=session,
        input_name=input_name,
        output_name=output_name,
        labels=labels,
        batch_size=batch_size,
        log=context.log,
    )


def _refuse_a_card_this_runtime_does_not_run(card: ModelCard | HeadCard) -> None:
    if not isinstance(card, HeadCard):
        raise ValueError(f"head runtime onnx runs a HeadCard, got a {type(card).__name__}")
    if card.runtime != RUNTIME:
        raise ValueError(
            f"head runtime onnx runs a card whose runtime is {RUNTIME!r}, "
            f"and this card's is {card.runtime!r}"
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
            f"head runtime onnx runs only on resource device {DEVICE!r}, got {device!r}"
        )


def _require_roles(files: Mapping[str, Path], registry: TaxonRegistry | None) -> None:
    for role in (GRAPH_ROLE, REGISTRY_ROLE):
        if role not in files:
            raise ValueError(
                f"head runtime onnx needs a file under the role {role!r}, "
                f"and got the roles {sorted(files)}"
            )
    if registry is None:
        raise ValueError(
            f"head runtime onnx needs the registry loaded from the role {REGISTRY_ROLE!r}, "
            "and was given none"
        )


def _only_node(nodes: Sequence[object], kind: str, width: int, source: str) -> str:
    """The name of the graph's one input or output, once it is float32 [batch, width]."""
    if len(nodes) != 1:
        raise ValueError(
            f"head runtime onnx needs a graph with one {kind}, and it has {len(nodes)} "
            f"{kind}s: {[node.name for node in nodes]}"
        )
    [node] = nodes
    if node.type != FLOAT32:
        raise ValueError(
            f"head runtime onnx needs the graph's {kind} {node.name!r} to be {FLOAT32!r}, "
            f"and it is {node.type!r}"
        )
    shape = list(node.shape)
    # onnxruntime gives a fixed dimension as an int, a symbolic one as a str, and an
    # unknown one as None. The batch must not be fixed, since the last batch is shorter.
    if len(shape) != 2 or type(shape[0]) is int or shape[1] != width:
        raise ValueError(
            f"head runtime onnx needs the graph's {kind} {node.name!r} to have shape "
            f"[batch, {width}], the batch not fixed and the width {source}, "
            f"and it is {shape}"
        )
    return node.name


class OnnxHead:
    """A head's loaded ONNX graph, run over one recording's embeddings at a time."""

    def __init__(
        self,
        *,
        session: object,
        input_name: str,
        output_name: str,
        labels: tuple[str, ...],
        batch_size: int,
        log: Log,
    ) -> None:
        self._session = session
        self._input_name = input_name
        self._output_name = output_name
        self._labels = labels
        self._batch_size = batch_size
        self._log = log

    def run(self, input: Input) -> Iterator[WindowOutput]:
        """Score the recording's vectors a batch at a time, yielding each window in order."""
        if not isinstance(input, Embeddings):
            raise TypeError(f"head runtime onnx runs on embeddings, got a {type(input).__name__}")
        return self._windows(input)

    def _windows(self, embeddings: Embeddings) -> Iterator[WindowOutput]:
        values = embeddings.values
        self._log(f"running {len(values)} windows of embeddings")
        for first in range(0, len(values), self._batch_size):
            # A slice of a C-contiguous float32 array, which onnxruntime takes as it is.
            batch = values[first : first + self._batch_size]
            [scores] = self._session.run([self._output_name], {self._input_name: batch})
            starts = embeddings.starts[first : first + len(batch)].tolist()
            ends = embeddings.ends[first : first + len(batch)].tolist()
            _refuse_scores_that_do_not_fit(scores, starts, self._labels)
            for start, end, row in zip(starts, ends, scores.tolist(), strict=True):
                yield WindowOutput(start=start, end=end, scores=self._scored(row))

    def _scored(self, scores: list[float]) -> tuple[ClassScore, ...]:
        return tuple(
            ClassScore(label=label, score=score)
            for label, score in zip(self._labels, scores, strict=True)
        )

    def after_recording(self) -> None:
        """Nothing to do: a recording leaves no files behind, and the graph stays loaded."""

    def clean_up(self) -> None:
        """Release the session."""
        self._session = None


def _refuse_scores_that_do_not_fit(
    scores: "np.ndarray", starts: Sequence[float], labels: Sequence[str]
) -> None:
    """Refuse a batch's output unless it is one finite score per window and label."""
    if scores.shape != (len(starts), len(labels)):
        raise ValueError(
            f"head runtime onnx gave scores of shape {scores.shape} for {len(starts)} "
            f"windows of {len(labels)} labels, which needs shape {(len(starts), len(labels))}"
        )
    found = np.argwhere(~np.isfinite(scores))
    if len(found):
        window, label = found[0]
        raise ValueError(
            f"head runtime onnx gave the score {float(scores[window, label])!r} for "
            f"{labels[label]!r} in the window at {starts[window]} s"
        )
