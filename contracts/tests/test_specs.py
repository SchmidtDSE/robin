import pytest
from pydantic import ValidationError

from robin_contracts.canonical import canonical_json_bytes
from robin_contracts.cards import BackendResampled, ModelCard, ModelRef, RunnerResampled, model_ref
from robin_contracts.embedding_transforms import Identity, L2Norm
from robin_contracts.specs import (
    AudioSpec,
    Recipe,
    WindowGeometry,
    recipe,
    window_bounds,
    window_count,
)

MODEL = ModelRef(name="owl", version="1", digest="sha256:v1:" + "0" * 64)


def build_audio(**overrides) -> AudioSpec:
    fields = {
        "sample_rate": 32000,
        "window_duration": 3.0,
        "window_overlap": 0.0,
        "downmix": "mean",
        "resampler": RunnerResampled(algorithm="soxr_hq"),
        "pad": "time_scaled",
    }
    return AudioSpec(**(fields | overrides))


def build_recipe(**overrides) -> Recipe:
    fields = {
        "model": MODEL,
        "backend": "tensorflow",
        "audio": build_audio(),
        "embedding_transform": L2Norm(),
        "dtype": "float32",
    }
    return Recipe(**(fields | overrides))


def build_card(**overrides) -> ModelCard:
    fields = {
        "model_name": "owl",
        "model_version": "1",
        "runtime": "tensorflow",
        "window_duration": 3.0,
        "window_overlap": 1.5,
        "sample_rate": 32000,
        "min_detection_threshold": 0.0,
        "score_domain": "probability",
        "audio": {
            "downmix": "first",
            "resampler": {"by": "runner", "algorithm": "librosa"},
            "pad": "drop",
        },
        "backend": "tensorflow",
        "embedding_transform": {"kind": "l2"},
        "dtype": "float16",
    }
    return ModelCard.model_validate(fields | overrides)


def test_the_recipe_a_card_states_takes_every_fact_from_the_card():
    card = build_card()

    assert recipe(card) == Recipe(
        model=model_ref(card),
        backend="tensorflow",
        audio=AudioSpec(
            sample_rate=32000,
            window_duration=3.0,
            window_overlap=1.5,
            downmix="first",
            resampler=RunnerResampled(algorithm="librosa"),
            pad="drop",
        ),
        embedding_transform=L2Norm(),
        dtype="float16",
    )


def test_the_recipe_of_a_card_whose_library_resamples_says_so():
    resampler = {"by": "backend", "library": "birdnet", "version": "2.4"}
    card = build_card(audio={"downmix": "mean", "resampler": resampler, "pad": "drop"})

    assert recipe(card).audio.resampler == BackendResampled(library="birdnet", version="2.4")


def test_two_cards_that_differ_have_recipes_that_differ():
    assert recipe(build_card()).id != recipe(build_card(window_overlap=0.0)).id


def test_recipe_fingerprint_is_a_full_length_versioned_digest():
    fingerprint = build_recipe().id

    assert fingerprint.startswith("sha256:v1:")
    digest = fingerprint.removeprefix("sha256:v1:")
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_equal_recipes_have_equal_fingerprints():
    assert build_recipe().id == build_recipe().id


def test_recipes_differing_only_in_embedding_transform_have_different_fingerprints():
    one = build_recipe(embedding_transform=L2Norm())
    other = build_recipe(embedding_transform=Identity())

    assert one.id != other.id


def test_recipes_differing_only_in_dtype_have_different_fingerprints():
    assert build_recipe(dtype="float32").id != build_recipe(dtype="float16").id


def test_recipe_canonical_bytes_are_stable_across_constructions():
    one = build_recipe(audio=build_audio(resampler=BackendResampled(library="birdnet", version="2")))
    other = build_recipe(audio=build_audio(resampler=BackendResampled(library="birdnet", version="2")))

    assert canonical_json_bytes(one) == canonical_json_bytes(other)


def test_audio_spec_geometry_carries_window_overlap_and_pad():
    geometry = build_audio(window_duration=3.0, window_overlap=2.0, pad="drop").geometry

    assert geometry == WindowGeometry(window_duration=3.0, window_overlap=2.0, pad="drop")


NONFINITE = [float("nan"), float("inf"), float("-inf")]
BAD_WINDOW_VALUES = [("window_duration", v) for v in [0.0, -1.0, *NONFINITE]] + [
    ("window_overlap", v) for v in [-1.0, *NONFINITE]
]


@pytest.mark.parametrize(("field", "value"), BAD_WINDOW_VALUES)
def test_window_geometry_rejects_a_bad_duration_or_overlap(field, value):
    fields = {"window_duration": 3.0, "window_overlap": 0.0, "pad": "drop"}

    with pytest.raises(ValidationError) as exc:
        WindowGeometry(**(fields | {field: value}))

    assert exc.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize(("field", "value"), BAD_WINDOW_VALUES)
def test_audio_spec_rejects_a_bad_duration_or_overlap(field, value):
    with pytest.raises(ValidationError) as exc:
        build_audio(**{field: value})

    assert exc.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize("overlap", [3.0, 4.0])
def test_windows_that_would_not_advance_are_refused(overlap):
    with pytest.raises(ValidationError, match="must be less than window_duration"):
        WindowGeometry(window_duration=3.0, window_overlap=overlap, pad="drop")
    with pytest.raises(ValidationError, match="must be less than window_duration"):
        build_audio(window_duration=3.0, window_overlap=overlap)


def test_each_window_starts_one_duration_less_the_overlap_after_the_last():
    geometry = WindowGeometry(window_duration=3.0, window_overlap=1.0, pad="drop")

    assert geometry.hop == 2.0


def test_window_count_counts_whole_windows_at_a_hop():
    geometry = WindowGeometry(window_duration=3.0, window_overlap=0.0, pad="drop")

    assert window_count(30.0, geometry) == 10


def test_window_count_is_zero_for_audio_shorter_than_one_window_when_dropping():
    geometry = WindowGeometry(window_duration=3.0, window_overlap=0.0, pad="drop")

    assert window_count(1.5, geometry) == 0


@pytest.mark.parametrize("pad", ["centre_crop_end_pad", "time_scaled"])
def test_window_count_is_one_for_audio_shorter_than_one_window_when_padding(pad):
    assert window_count(1.5, WindowGeometry(window_duration=3.0, window_overlap=0.0, pad=pad)) == 1


@pytest.mark.parametrize(
    ("pad", "expected"),
    [("drop", 3), ("centre_crop_end_pad", 4), ("time_scaled", 4)],
)
def test_window_count_drops_a_trailing_partial_window_and_pads_it_under_the_other_policies(
    pad, expected
):
    geometry = WindowGeometry(window_duration=3.0, window_overlap=0.0, pad=pad)

    assert window_count(10.0, geometry) == expected


# Each duration is a whole number of hops past the window on paper and a hair off it
# in binary floating point: three hops lands just under the boundary and seven just
# over, so a count without a tolerance is wrong on one side or the other.
@pytest.mark.parametrize(("hops", "expected"), [(3, 4), (7, 8)])
@pytest.mark.parametrize("pad", ["drop", "centre_crop_end_pad", "time_scaled"])
def test_window_count_is_stable_at_an_exact_window_boundary(pad, hops, expected):
    duration = 3.0 + sum([0.1] * hops)

    geometry = WindowGeometry(window_duration=3.0, window_overlap=2.9, pad=pad)

    assert window_count(duration, geometry) == expected


def test_window_bounds_step_by_the_hop_and_span_one_window():
    geometry = WindowGeometry(window_duration=3.0, window_overlap=2.0, pad="drop")

    assert window_bounds(5.0, geometry) == [(0.0, 3.0), (1.0, 4.0), (2.0, 5.0)]


@pytest.mark.parametrize("pad", ["drop", "centre_crop_end_pad", "time_scaled"])
def test_window_bounds_give_one_window_per_counted_window(pad):
    geometry = WindowGeometry(window_duration=3.0, window_overlap=0.0, pad=pad)

    assert len(window_bounds(10.0, geometry)) == window_count(10.0, geometry)
