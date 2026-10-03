import math

import numpy as np
import pytest

from robin_contracts.cards import AudioGeometry, ModelCard, RunnerResampled
from robin_contracts.output_contracts import ScoresRequest
from robin_contracts.records import ClassScore, WindowOutput
from robin_contracts.registry import RegistryEntry, TaxonRegistry
from robin_contracts.specs import WindowGeometry, window_bounds
from robin_contracts.work import RecordingRef
from robin_inference_engine import errors
from robin_inference_engine.accept_window import AcceptanceBoundary

REGISTRY_DIGEST = "sha256:" + "a" * 64

FINGERPRINT = "sha256:" + "0" * 64

# The model's own floor: its library returns only the scores at or above this.
MODEL_FLOOR = 0.01


def build_scores_request(**overrides) -> ScoresRequest:
    fields = {
        "contract_id": "robin.scores.arrow/1",
        "retention": "thresholded",
        "min_score": MODEL_FLOOR,
    }
    return ScoresRequest(**(fields | overrides))


FULL = build_scores_request(retention="full", min_score=None)


def build_registry(*labels: str) -> TaxonRegistry:
    entries = tuple(
        RegistryEntry(class_index=index, label=label, label_kind="non_taxonomic")
        for index, label in enumerate(labels)
    )
    return TaxonRegistry(fingerprint=FINGERPRINT, entries=entries)


def build_geometry() -> WindowGeometry:
    return WindowGeometry(window_duration=3.0, window_overlap=0.0, pad="time_scaled")


def build_card(*, geometry: WindowGeometry | None = None, **overrides) -> ModelCard:
    geometry = geometry or build_geometry()
    fields = {
        "model_name": "test-model",
        "model_version": "1",
        "runtime": "none",
        "window_duration": geometry.window_duration,
        "window_overlap": geometry.window_overlap,
        "sample_rate": 16000,
        "min_detection_threshold": MODEL_FLOOR,
        "score_domain": "probability",
        "taxa_registry_digest": REGISTRY_DIGEST,
        "audio": AudioGeometry(
            downmix="mean", resampler=RunnerResampled(algorithm="soxr_hq"), pad=geometry.pad
        ),
        "backend": "none",
        "dtype": "float32",
        "can_emit_embeddings": True,
        "embedding_dim": 4,
        "embedding_dtype": "float32",
    }
    return ModelCard(**(fields | overrides))


# A model that emits every label for every window.
FULL_STREAM_CARD = build_card(min_detection_threshold=0.0)


def build_recordings(*durations: float | None) -> tuple[RecordingRef, ...]:
    return tuple(
        RecordingRef(
            namespace="soundhub",
            value=f"rec-{position}",
            audio_uri=f"s3://b/{position}.wav",
            duration_seconds=duration,
        )
        for position, duration in enumerate(durations)
    )


RECORDINGS = build_recordings(30.0, 30.0)


def build_boundary(
    *, opened: int | None = 0, geometry: WindowGeometry | None = None, **overrides
) -> AcceptanceBoundary:
    geometry = geometry or build_geometry()
    fields = {
        "geometry": geometry,
        "card": build_card(geometry=geometry),
        "recordings": RECORDINGS,
        "registry": build_registry("rain", "wind"),
        "scores": build_scores_request(),
    }
    boundary = AcceptanceBoundary(**(fields | overrides))
    if opened is not None:
        boundary.begin_recording(opened)
    return boundary


def build_window(**overrides) -> WindowOutput:
    fields = {
        "start": 0.0,
        "end": 3.0,
        "scores": (ClassScore(label="rain", score=0.5),),
    }
    return WindowOutput(**(fields | overrides))


def build_embedding(*values: float, dtype=np.float32) -> np.ndarray:
    return np.array(values or (0.1, 0.2, 0.3, 0.4), dtype=dtype)


def test_accept_returns_the_window_it_was_given():
    accepted = build_boundary().accept(build_window(start=3.0, end=6.0))

    assert accepted.recording is RECORDINGS[0]
    assert accepted.start == 3.0
    assert accepted.end == 6.0
    assert accepted.scores == (ClassScore(label="rain", score=0.5),)
    assert accepted.embedding is None


def test_a_window_belongs_to_the_recording_that_is_open():
    boundary = build_boundary(opened=0)
    first = boundary.accept(build_window(start=3.0, end=6.0))
    boundary.begin_recording(1)
    # The second recording starts its own ordering, so an earlier start is accepted.
    second = boundary.accept(build_window(start=0.0, end=3.0))

    assert (first.recording.namespace, first.recording.value) == ("soundhub", "rec-0")
    assert (second.recording.namespace, second.recording.value) == ("soundhub", "rec-1")


def test_a_refusal_names_the_open_recording():
    boundary = build_boundary(opened=1)

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(build_window(start=1.0, end=4.0))

    assert exc.value.stage == "accept_window"
    assert exc.value.recording is RECORDINGS[1]


@pytest.mark.parametrize("position", [2, 7, -1])
def test_begin_recording_refuses_a_position_outside_the_work(position):
    boundary = build_boundary(opened=None)

    with pytest.raises(RuntimeError, match="not one of this work's"):
        boundary.begin_recording(position)


def test_begin_recording_refuses_a_recording_already_processed():
    boundary = build_boundary(opened=None)
    boundary.begin_recording(0)
    boundary.begin_recording(1)

    with pytest.raises(RuntimeError, match="already processed"):
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
    geometry = WindowGeometry(window_duration=3.0, window_overlap=2.0, pad="time_scaled")
    boundary = build_boundary(geometry=geometry)

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
def test_accept_refuses_a_start_rounded_to_one_decimal(start):
    with pytest.raises(errors.EngineError) as exc:
        build_boundary().accept(build_window(start=start, end=start + 3.0))

    assert exc.value.code == errors.WINDOW_OFF_GEOMETRY


def test_accept_refuses_a_start_one_sample_off_the_grid_at_96_khz():
    start = 3.0 + 1 / 96000

    with pytest.raises(errors.EngineError) as exc:
        build_boundary().accept(build_window(start=start, end=start + 3.0))

    assert exc.value.code == errors.WINDOW_OFF_GEOMETRY


def test_accept_refuses_a_window_of_the_wrong_duration():
    with pytest.raises(errors.EngineError) as exc:
        build_boundary().accept(build_window(start=3.0, end=8.0))

    assert exc.value.code == errors.WINDOW_OFF_GEOMETRY


def test_accept_refuses_a_window_slightly_longer_than_the_recipe_declares():
    with pytest.raises(errors.EngineError) as exc:
        build_boundary().accept(build_window(start=3.0, end=6.08))

    assert exc.value.code == errors.WINDOW_OFF_GEOMETRY


def test_accept_takes_every_window_of_a_day_long_recording_at_a_hop_with_no_exact_binary_form():
    # An 11.1 s hop has no exact binary form, so i * hop drifts off the grid by a few
    # trillionths of a second over a day of windows.
    duration = 24 * 60 * 60.0
    geometry = WindowGeometry(window_duration=12.0, window_overlap=0.9, pad="time_scaled")
    boundary = build_boundary(geometry=geometry, recordings=build_recordings(duration))

    bounds = window_bounds(duration, geometry)
    accepted = [boundary.accept(build_window(start=start, end=end)) for start, end in bounds]

    assert [(window.start, window.end) for window in accepted] == bounds


@pytest.mark.parametrize("start", [9.0, 12.0])
def test_accept_refuses_a_start_at_or_past_the_recording_duration(start):
    boundary = build_boundary(recordings=build_recordings(9.0))

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(build_window(start=start, end=start + 3.0))

    assert exc.value.code == errors.WINDOW_OUTSIDE_RECORDING


@pytest.mark.parametrize("duration", [10.0, None])
@pytest.mark.parametrize("start", [-6.0, -3.0, -1e-12])
def test_accept_refuses_a_negative_start_even_on_the_hop_grid(start, duration):
    boundary = build_boundary(recordings=build_recordings(duration))

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(build_window(start=start, end=start + 3.0))

    assert exc.value.code == errors.WINDOW_OUTSIDE_RECORDING


@pytest.mark.parametrize(("duration", "start"), [(1.0, 0.0), (10.0, 9.0)])
def test_accept_refuses_a_partial_window_under_drop_policy(duration, start):
    boundary = build_boundary(
        geometry=WindowGeometry(window_duration=3.0, window_overlap=0.0, pad="drop"),
        recordings=build_recordings(duration),
    )

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(build_window(start=start, end=start + 3.0))

    assert exc.value.code == errors.WINDOW_OUTSIDE_RECORDING


def test_accept_allows_a_complete_window_under_drop_policy():
    boundary = build_boundary(
        geometry=WindowGeometry(window_duration=3.0, window_overlap=0.0, pad="drop"),
        recordings=build_recordings(3.0),
    )

    accepted = boundary.accept(build_window(start=0.0, end=3.0))

    assert (accepted.start, accepted.end) == (0.0, 3.0)


@pytest.mark.parametrize("pad", ["centre_crop_end_pad", "time_scaled"])
def test_accept_allows_a_trailing_pad_window_past_the_audio(pad):
    geometry = WindowGeometry(window_duration=3.0, window_overlap=2.0, pad=pad)

    accepted = build_boundary(geometry=geometry, recordings=build_recordings(9.05)).accept(
        build_window(start=9.0, end=12.0)
    )

    assert accepted.end == 12.0


@pytest.mark.parametrize("pad", ["centre_crop_end_pad", "time_scaled"])
def test_accept_allows_a_padded_last_window_starting_just_before_the_end(pad):
    geometry = WindowGeometry(window_duration=3.0, window_overlap=0.0, pad=pad)
    boundary = build_boundary(geometry=geometry, recordings=build_recordings(9.000001))

    accepted = boundary.accept(build_window(start=9.0, end=12.0))

    assert (accepted.start, accepted.end) == (9.0, 12.0)


@pytest.mark.parametrize("pad", ["drop", "centre_crop_end_pad", "time_scaled"])
def test_accept_checks_no_duration_when_the_recording_declares_none(pad):
    accepted = build_boundary(
        geometry=WindowGeometry(window_duration=3.0, window_overlap=0.0, pad=pad),
        recordings=build_recordings(None),
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
    boundary = build_boundary(card=FULL_STREAM_CARD, scores=FULL)

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(build_window(scores=scores))

    assert exc.value.code == errors.INCOMPLETE_FULL_SCORES


def test_accept_allows_empty_scores_under_reduced_retention():
    accepted = build_boundary().accept(build_window(scores=()))

    assert accepted.scores == ()


def test_accept_refuses_an_embedding_that_was_not_requested():
    boundary = build_boundary(expect_embeddings=False)

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(build_window(embedding=build_embedding()))

    assert exc.value.code == errors.UNEXPECTED_EMBEDDING


@pytest.mark.parametrize(
    ("embedding", "fragment"),
    [
        (np.zeros((2, 2), dtype=np.float32), "1-D"),
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


@pytest.mark.parametrize(
    ("declared", "arriving"),
    [("float32", np.float16), ("float32", np.float64), ("float16", np.float32)],
)
def test_accept_refuses_an_embedding_that_is_not_the_declared_dtype(declared, arriving):
    boundary = build_boundary(
        expect_embeddings=True,
        card=build_card(embedding_dtype=declared),
    )

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(build_window(embedding=build_embedding(dtype=arriving)))

    assert exc.value.code == errors.MALFORMED_EMBEDDING
    assert declared in exc.value.detail
    assert np.dtype(arriving).name in exc.value.detail


@pytest.mark.parametrize("declared", ["float16", "float32"])
def test_accept_takes_an_embedding_at_the_dtype_the_card_declares(declared):
    boundary = build_boundary(
        expect_embeddings=True,
        card=build_card(embedding_dtype=declared),
    )

    accepted = boundary.accept(
        build_window(embedding=build_embedding(dtype=np.dtype(declared)))
    )

    assert accepted.embedding.dtype == np.dtype(declared)


def test_accept_refuses_an_embedding_that_is_not_in_this_machines_byte_order():
    # The comparison is between dtypes, not between their names: a big-endian array
    # reports the name "float32", is 1-D, C-contiguous, finite and the right width,
    # and narrows through astype without complaint, so nothing downstream catches it.
    # Comparing names here would let it reach storage as raw bytes in the wrong order.
    boundary = build_boundary(expect_embeddings=True)
    swapped = np.arange(4, dtype=">f4")
    assert swapped.dtype.name == "float32"

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(build_window(embedding=swapped))

    assert exc.value.code == errors.MALFORMED_EMBEDDING


@pytest.mark.parametrize(("source", "storage"), [("float32", "float16"), ("float16", "float32")])
def test_an_embedding_is_matched_against_the_emitted_precision_not_the_stored_one(source, storage):
    boundary = build_boundary(
        expect_embeddings=True, card=build_card(embedding_dtype=source, dtype=storage)
    )

    accepted = boundary.accept(build_window(embedding=build_embedding(dtype=np.dtype(source))))
    assert accepted.embedding.dtype == np.dtype(source)

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(
            build_window(start=3.0, end=6.0, embedding=build_embedding(dtype=np.dtype(storage)))
        )
    assert exc.value.code == errors.MALFORMED_EMBEDDING


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


# --- Scores against the request -------------------------------------------------

UNREQUESTED_SCORES = {
    "out_of_domain": (ClassScore(label="rain", score=math.nan),),
    "unknown_label": (ClassScore(label="sleet", score=0.5),),
    "duplicate_label": (ClassScore(label="rain", score=0.5), ClassScore(label="rain", score=0.4)),
}


@pytest.mark.parametrize("scores", UNREQUESTED_SCORES.values(), ids=UNREQUESTED_SCORES.keys())
def test_scores_a_work_did_not_request_are_dropped_unchecked(scores):
    accepted = build_boundary(scores=None).accept(build_window(scores=scores))

    assert accepted.scores == ()


def test_scores_a_work_did_not_request_need_no_registry():
    accepted = build_boundary(scores=None, registry=None).accept(build_window())

    assert accepted.scores == ()


REQUESTS = {
    "full": FULL,
    "thresholded": build_scores_request(min_score=0.5),
    "top_k": build_scores_request(retention="top_k", min_score=0.5, top_k=1),
}


@pytest.mark.parametrize("request_", REQUESTS.values(), ids=REQUESTS.keys())
def test_a_full_stream_missing_a_label_is_refused_whatever_was_requested(request_):
    boundary = build_boundary(card=FULL_STREAM_CARD, scores=request_)

    with pytest.raises(errors.EngineError) as exc:
        boundary.accept(build_window(scores=(ClassScore(label="rain", score=0.9),)))

    assert exc.value.code == errors.INCOMPLETE_FULL_SCORES


@pytest.mark.parametrize("request_", REQUESTS.values(), ids=REQUESTS.keys())
def test_a_full_stream_is_accepted_whole_whatever_was_requested(request_):
    # The request's floor and k are applied after the boundary, not checked by it.
    scores = (ClassScore(label="rain", score=0.9), ClassScore(label="wind", score=0.0))
    boundary = build_boundary(card=FULL_STREAM_CARD, scores=request_)

    assert boundary.accept(build_window(scores=scores)).scores == scores


@pytest.mark.parametrize("retention", ["thresholded", "top_k"])
def test_a_thresholded_stream_carrying_a_score_below_the_models_floor_is_refused(retention):
    request = build_scores_request(
        retention=retention, min_score=MODEL_FLOOR, top_k=2 if retention == "top_k" else None
    )
    below = (ClassScore(label="rain", score=0.009),)

    with pytest.raises(errors.EngineError) as exc:
        build_boundary(scores=request).accept(build_window(start=3.0, end=6.0, scores=below))

    assert exc.value.code == errors.SCORE_BELOW_FLOOR
    assert exc.value.recording is RECORDINGS[0]
    assert exc.value.window_start_s == 3.0
    assert "0.009" in exc.value.detail and str(MODEL_FLOOR) in exc.value.detail


def test_a_thresholded_stream_may_carry_a_score_exactly_at_the_models_floor():
    at_floor = (ClassScore(label="rain", score=MODEL_FLOOR),)

    assert build_boundary().accept(build_window(scores=at_floor)).scores == at_floor


def test_a_thresholded_stream_is_not_checked_against_the_requests_floor_or_k():
    request = build_scores_request(retention="top_k", min_score=0.5, top_k=1)
    scores = (ClassScore(label="rain", score=0.2), ClassScore(label="wind", score=0.1))

    assert build_boundary(scores=request).accept(build_window(scores=scores)).scores == scores


def test_a_non_finite_score_is_out_of_domain_before_it_meets_the_floor():
    with pytest.raises(errors.EngineError) as exc:
        build_boundary().accept(build_window(scores=(ClassScore(label="rain", score=math.nan),)))

    assert exc.value.code == errors.SCORE_OUT_OF_DOMAIN
