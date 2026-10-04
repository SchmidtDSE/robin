"""Reading a head's input: one recording's embeddings file, checked against the head card
and the recording before any of its rows is read."""

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import TypeVar

import numpy as np
import pyarrow as pa

from robin_contracts.cards import HeadCard
from robin_contracts.inputs import Embeddings
from robin_contracts.work import RecordingRef
from robin_inference_engine import errors
from robin_inference_engine.artifacts.embeddings import read_embeddings
from robin_inference_engine.artifacts.metadata import (
    BACKBONE_CARD_DIGEST_KEY,
    BACKBONE_REF_KEY,
    EMBEDDING_DIM_KEY,
    RECORDING_NAMESPACE_KEY,
    RECORDING_VALUE_KEY,
)

T = TypeVar("T")


def verify_head_input(
    header: Mapping[str, str], card: HeadCard, recording: RecordingRef
) -> None:
    """Refuse a file that another backbone than the head's wrote, that is not the width
    the head takes, or that holds another recording than `recording`."""
    head = f"head {card.model_name}/{card.model_version}"
    _require(
        header,
        BACKBONE_REF_KEY,
        card.backbone.id,
        f"{head} reads backbone",
        errors.HEAD_INPUT_BACKBONE_MISMATCH,
        recording,
    )
    _require(
        header,
        BACKBONE_CARD_DIGEST_KEY,
        card.backbone.digest,
        f"{head} reads the backbone card",
        errors.HEAD_INPUT_BACKBONE_MISMATCH,
        recording,
    )
    _require(
        header,
        EMBEDDING_DIM_KEY,
        str(card.embedding_dim),
        f"{head} takes embedding_dim",
        errors.HEAD_INPUT_WIDTH_MISMATCH,
        recording,
    )
    _require(
        header,
        RECORDING_NAMESPACE_KEY,
        recording.namespace,
        "the work pairs it with namespace",
        errors.HEAD_INPUT_RECORDING_MISMATCH,
        recording,
    )
    _require(
        header,
        RECORDING_VALUE_KEY,
        recording.value,
        "the work pairs it with value",
        errors.HEAD_INPUT_RECORDING_MISMATCH,
        recording,
    )


def read_head_input(path: Path, *, card: HeadCard, recording: RecordingRef) -> Embeddings:
    """The recording's vectors as stored, widened to float32, one row per window.

    The file's bytes, schema and header are checked before any row is read.
    """
    checksum = recording.embeddings.checksum
    with ExitStack() as files:
        stream = _attributed(
            lambda: files.enter_context(read_embeddings(path, expected_checksum=checksum)),
            recording,
        )
        verify_head_input(stream.metadata, card, recording)
        batches = _attributed(lambda: list(stream.batches), recording)
    return _rows(batches, card.embedding_dim)


def _require(
    header: Mapping[str, str],
    key: str,
    expected: str,
    source: str,
    code: str,
    recording: RecordingRef,
) -> None:
    declared = header.get(key)
    if declared != expected:
        raise errors.EngineError(
            code,
            errors.READ_INPUT_ARTIFACT,
            f"the embeddings file of recording {errors.named(recording)} declares {key} "
            f"{declared!r}, but {source} {expected!r}",
            recording=recording,
        )


def _attributed(read: Callable[[], T], recording: RecordingRef) -> T:
    """`read()`, with a failure the reader reports attributed to `recording`."""
    try:
        return read()
    except errors.EngineError as error:
        raise errors.EngineError(
            error.code,
            error.stage,
            f"the embeddings file of recording {errors.named(recording)}: {error.detail}",
            recording=recording,
        ) from error


def _rows(batches: list[pa.RecordBatch], dim: int) -> Embeddings:
    """Every row in order, copied into arrays the engine owns."""
    count = sum(batch.num_rows for batch in batches)
    starts = np.empty(count, dtype=np.float64)
    ends = np.empty(count, dtype=np.float64)
    values = np.empty((count, dim), dtype=np.float32)
    row = 0
    for batch in batches:
        end = row + batch.num_rows
        starts[row:end] = batch.column("window_start_s").to_numpy()
        ends[row:end] = batch.column("window_end_s").to_numpy()
        # flatten(), unlike .values, starts at a sliced batch's own offset. Assigning
        # float16 into the float32 array widens it exactly.
        flat = batch.column("embedding").flatten().to_numpy(zero_copy_only=False)
        values[row:end] = flat.reshape(-1, dim)
        row = end
    return Embeddings(starts=starts, ends=ends, values=values)
