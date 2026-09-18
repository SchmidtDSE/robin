import math

import numpy as np
import pytest

from robin_contracts.protocols import ModelCapabilities
from robin_contracts.records import ClassScore, WindowOutput
from robin_contracts.registry import RegistryEntry, TaxonRegistry
from robin_contracts.specs import WindowGeometry
from robin_inference_engine import errors
from robin_inference_engine.acceptance import AcceptanceBoundary

FINGERPRINT = "sha256:" + "0" * 64


def build_registry(*labels: str) -> TaxonRegistry:
    entries = tuple(
        RegistryEntry(class_index=index, label=label, label_kind="non_taxonomic")
        for index, label in enumerate(labels)
    )
    return TaxonRegistry(fingerprint=FINGERPRINT, entries=entries)


def build_capabilities(**overrides) -> ModelCapabilities:
    fields = {
        "emits_scores": True,
        "emits_embeddings": True,
        "score_domain": "probability",
        "embedding_dim": 4,
    }
    return ModelCapabilities(**(fields | overrides))


def build_boundary(*, opened: int | None = 0, **overrides) -> AcceptanceBoundary:
    fields = {
        "geometry": WindowGeometry(window=3.0, hop=3.0, pad="time_scaled"),
        "capabilities": build_capabilities(),
        "durations": {0: 30.0, 1: 30.0},
        "registry": build_registry("rain", "wind"),
    }
    boundary = AcceptanceBoundary(**(fields | overrides))
    if opened is not None:
        boundary.begin_recording(opened)
    return boundary


def build_window(**overrides) -> WindowOutput:
    fields = {
        "recording_index": 0,
        "start": 0.0,
        "end": 3.0,
        "scores": (ClassScore(label="rain", score=0.5),),
    }
    return WindowOutput(**(fields | overrides))


def build_embedding(*values: float) -> np.ndarray:
    return np.array(values or (0.1, 0.2, 0.3, 0.4), dtype=np.float32)


def test_accept_returns_the_window_it_was_given():
    accepted = build_boundary().accept(build_window(start=3.0, end=6.0))

    assert accepted.recording_index == 0
    assert accepted.start == 3.0
    assert accepted.end == 6.0
    assert accepted.scores == (ClassScore(label="rain", score=0.5),)
    assert accepted.embedding is None


def test_accept_refuses_a_recording_index_not_in_the_work():
    with pytest.raises(errors.EngineError) as exc:
        build_boundary().accept(build_window(recording_index=7))

    assert exc.value.code == errors.UNKNOWN_RECORDING_INDEX
    assert exc.value.stage == "accept_window"


def test_accept_refuses_a_window_for_a_recording_that_is_not_open():
    with pytest.raises(errors.EngineError) as exc:
        build_boundary(opened=0).accept(build_window(recording_index=1))

    assert exc.value.code == errors.WINDOW_OUT_OF_ORDER


def test_begin_recording_refuses_an_unknown_or_closed_recording():
    boundary = build_boundary(opened=None)

    with pytest.raises(RuntimeError):
        boundary.begin_recording(7)

    boundary.begin_recording(0)
    boundary.begin_recording(1)

    with pytest.raises(RuntimeError):
        boundary.begin_recording(0)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (math.nan, 3.0),
        (0.0, math.nan),
        (math.inf, 3.0),
        (0.0, math.inf),
        (-math.inf, 3.0),
    ],
)
def test_accept_refuses_a_non_finite_bound(start, end):
    with pytest.raises(errors.EngineError) as exc:
        build_boundary().accept(build_window(start=start, end=end))

    assert exc.value.code == errors.WINDOW_BOUNDS_INVALID


@pytest.mark.parametrize("end", [3.0, 2.0, -1.0])
def test_accept_refuses_an_end_at_or_before_its_start(end):
    with pytest.raises(errors.EngineError) as exc:
        build_boundary().accept(build_window(start=3.0, end=end))

    assert exc.value.code == errors.WINDOW_BOUNDS_INVALID


def test_accept_refuses_a_repeated_start_within_one_recording():
    boundary = build_boundary()
    boundary.accept(build_window(start=3.0, end=6.0))

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(build_window(start=3.0, end=6.0))

    assert exc.value.code == errors.DUPLICATE_WINDOW


def test_accept_refuses_a_start_below_the_previous_one():
    boundary = build_boundary()
    boundary.accept(build_window(start=3.0, end=6.0))

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(build_window(start=0.0, end=3.0))

    assert exc.value.code == errors.WINDOW_OUT_OF_ORDER


def test_accept_allows_overlapping_windows():
    boundary = build_boundary(geometry=WindowGeometry(window=3.0, hop=1.0, pad="time_scaled"))

    accepted = [
        boundary.accept(build_window(start=start, end=start + 3.0))
        for start in (0.0, 1.0, 2.0)
    ]

    assert [window.start for window in accepted] == [0.0, 1.0, 2.0]


def test_accept_refuses_a_start_off_the_hop_grid():
    with pytest.raises(errors.EngineError) as exc:
        build_boundary().accept(build_window(start=1.5, end=4.5))

    assert exc.value.code == errors.WINDOW_OFF_GEOMETRY


@pytest.mark.parametrize("start", [2.96, 3.04])
def test_accept_tolerates_a_start_rounded_to_one_decimal(start):
    accepted = build_boundary().accept(build_window(start=start, end=start + 3.0))

    assert accepted.start == start


def test_accept_refuses_a_window_of_the_wrong_duration():
    with pytest.raises(errors.EngineError) as exc:
        build_boundary().accept(build_window(start=3.0, end=8.0))

    assert exc.value.code == errors.WINDOW_OFF_GEOMETRY


@pytest.mark.parametrize("start", [9.0, 12.0])
def test_accept_refuses_a_start_at_or_past_the_recording_duration(start):
    boundary = build_boundary(durations={0: 9.0})

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(build_window(start=start, end=start + 3.0))

    assert exc.value.code == errors.WINDOW_OUTSIDE_RECORDING


@pytest.mark.parametrize("duration", [10.0, None])
@pytest.mark.parametrize("start", [-6.0, -3.0, -0.01])
def test_accept_refuses_a_negative_start_even_on_the_hop_grid(start, duration):
    boundary = build_boundary(durations={0: duration})

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(build_window(start=start, end=start + 3.0))

    assert exc.value.code == errors.WINDOW_OUTSIDE_RECORDING


@pytest.mark.parametrize(("duration", "start"), [(1.0, 0.0), (10.0, 9.0)])
def test_accept_refuses_a_partial_window_under_drop_policy(duration, start):
    boundary = build_boundary(
        geometry=WindowGeometry(window=3.0, hop=3.0, pad="drop"),
        durations={0: duration},
    )

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(build_window(start=start, end=start + 3.0))

    assert exc.value.code == errors.WINDOW_OUTSIDE_RECORDING


def test_accept_allows_a_complete_window_under_drop_policy():
    boundary = build_boundary(
        geometry=WindowGeometry(window=3.0, hop=3.0, pad="drop"),
        durations={0: 3.0},
    )

    accepted = boundary.accept(build_window(start=0.0, end=3.0))

    assert (accepted.start, accepted.end) == (0.0, 3.0)


@pytest.mark.parametrize("pad", ["centre_crop_end_pad", "time_scaled"])
def test_accept_allows_a_trailing_pad_window_past_the_audio(pad):
    # The refused half is a 0.05 s band: the duration check runs after the geometry
    # check has already pinned end - start to the window, so a duration sitting a hair
    # above a hop multiple is the only shape that reaches it.
    geometry = WindowGeometry(window=3.0, hop=1.0, pad=pad)

    accepted = build_boundary(geometry=geometry, durations={0: 9.05}).accept(
        build_window(start=9.0, end=12.0)
    )

    assert accepted.end == 12.0

    with pytest.raises(errors.EngineError) as exc:
        build_boundary(geometry=geometry, durations={0: 9.05}).accept(
            build_window(start=9.0, end=12.08)
        )

    assert exc.value.code == errors.WINDOW_OUTSIDE_RECORDING


@pytest.mark.parametrize("pad", ["drop", "centre_crop_end_pad", "time_scaled"])
def test_accept_checks_no_duration_when_the_recording_declares_none(pad):
    accepted = build_boundary(
        geometry=WindowGeometry(window=3.0, hop=3.0, pad=pad),
        durations={0: None},
    ).accept(build_window(start=90.0, end=93.0))

    assert accepted.start == 90.0


def test_accept_refuses_a_window_before_any_recording_is_open():
    with pytest.raises(RuntimeError):
        build_boundary(opened=None).accept(build_window())


def test_a_refused_window_does_not_advance_the_accepted_start():
    boundary = build_boundary()
    boundary.accept(build_window(start=0.0, end=3.0))

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(
            build_window(start=3.0, end=6.0, scores=(ClassScore(label="rain", score=2.0),))
        )

    assert exc.value.code == errors.SCORE_OUT_OF_DOMAIN
    assert boundary.accept(build_window(start=3.0, end=6.0)).start == 3.0


@pytest.mark.parametrize("score", [2.0, -0.1, math.nan, math.inf, -math.inf])
def test_accept_refuses_a_score_outside_the_declared_domain(score):
    with pytest.raises(errors.EngineError) as exc:
        build_boundary().accept(build_window(scores=(ClassScore(label="rain", score=score),)))

    assert exc.value.code == errors.SCORE_OUT_OF_DOMAIN


def test_accept_refuses_a_label_the_registry_does_not_declare():
    with pytest.raises(errors.EngineError) as exc:
        build_boundary().accept(build_window(scores=(ClassScore(label="sleet", score=0.5),)))

    assert exc.value.code == errors.UNKNOWN_LABEL
    assert "sleet" in exc.value.detail
    assert FINGERPRINT in exc.value.detail


def test_accept_refuses_an_empty_label():
    with pytest.raises(errors.EngineError) as exc:
        build_boundary().accept(build_window(scores=(ClassScore(label="", score=0.5),)))

    assert exc.value.code == errors.UNKNOWN_LABEL


def test_accept_refuses_a_label_repeated_within_one_window():
    scores = (ClassScore(label="rain", score=0.5), ClassScore(label="rain", score=0.4))

    with pytest.raises(errors.EngineError) as exc:
        build_boundary().accept(build_window(scores=scores))

    assert exc.value.code == errors.DUPLICATE_LABEL


@pytest.mark.parametrize("scores", [(ClassScore(label="rain", score=0.5),), ()])
def test_accept_refuses_a_window_missing_a_label_under_full_retention(scores):
    boundary = build_boundary(require_full_scores=True)

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(build_window(scores=scores))

    assert exc.value.code == errors.INCOMPLETE_FULL_SCORES


def test_accept_allows_empty_scores_under_reduced_retention():
    accepted = build_boundary(require_full_scores=False).accept(build_window(scores=()))

    assert accepted.scores == ()


def test_accept_refuses_any_score_when_the_model_declares_no_registry():
    with pytest.raises(errors.EngineError) as exc:
        build_boundary(registry=None).accept(build_window())

    assert exc.value.code == errors.UNKNOWN_LABEL
    assert "registry" in exc.value.detail


@pytest.mark.parametrize(
    ("expect_embeddings", "emits_embeddings"),
    [(False, True), (True, False)],
)
def test_accept_refuses_an_embedding_that_was_not_requested(expect_embeddings, emits_embeddings):
    boundary = build_boundary(
        expect_embeddings=expect_embeddings,
        capabilities=build_capabilities(emits_embeddings=emits_embeddings),
    )

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(build_window(embedding=build_embedding()))

    assert exc.value.code == errors.UNEXPECTED_EMBEDDING


@pytest.mark.parametrize(
    ("embedding", "fragment"),
    [
        (np.zeros((2, 2), dtype=np.float32), "1-D"),
        (np.zeros(4), "float64"),
        (np.zeros(8, dtype=np.float32)[::2], "C-contiguous"),
        (np.zeros(3, dtype=np.float32), "4 values wide"),
    ],
)
def test_accept_refuses_a_malformed_embedding(embedding, fragment):
    boundary = build_boundary(expect_embeddings=True)

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(build_window(embedding=embedding))

    assert exc.value.code == errors.MALFORMED_EMBEDDING
    assert fragment in exc.value.detail


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_accept_refuses_a_non_finite_embedding_value(bad):
    boundary = build_boundary(expect_embeddings=True)

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(build_window(embedding=build_embedding(0.1, bad, 0.3, 0.4)))

    assert exc.value.code == errors.EMBEDDING_NOT_FINITE


def test_accept_writes_an_all_zero_embedding_through_unchanged():
    boundary = build_boundary(expect_embeddings=True)

    accepted = boundary.accept(build_window(embedding=np.zeros(4, dtype=np.float32)))

    assert np.array_equal(accepted.embedding, np.zeros(4, dtype=np.float32))


def test_accept_copies_the_embedding_so_a_reused_buffer_cannot_corrupt_it():
    boundary = build_boundary(expect_embeddings=True)
    buffer = build_embedding()
    window = build_window(embedding=buffer)

    accepted = boundary.accept(window)
    buffer[:] = 9.0

    assert accepted.embedding is not window.embedding
    assert np.array_equal(accepted.embedding, np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))


def test_accept_does_not_make_the_adapters_array_read_only():
    boundary = build_boundary(expect_embeddings=True)
    buffer = build_embedding()

    boundary.accept(build_window(embedding=buffer))

    assert buffer.flags["WRITEABLE"]


def test_accept_refuses_a_start_off_the_grid_on_the_negative_side():
    with pytest.raises(errors.EngineError) as exc:
        build_boundary().accept(build_window(start=-0.2, end=2.8))

    assert exc.value.code == errors.WINDOW_OFF_GEOMETRY


def test_full_scores_without_a_registry_is_an_engine_defect():
    with pytest.raises(RuntimeError):
        build_boundary(opened=None, registry=None, require_full_scores=True)
