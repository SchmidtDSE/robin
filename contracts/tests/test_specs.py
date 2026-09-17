import pytest

from robin_contracts.canonical import canonical_json_bytes
from robin_contracts.cards import ModelRef
from robin_contracts.embedding_transforms import Identity, L2Norm
from robin_contracts.specs import (
    AudioSpec,
    BackendResampled,
    Recipe,
    RunnerResampled,
    WindowGeometry,
    window_count,
)

MODEL = ModelRef(name="owl", version="1", digest="sha256:v1:" + "0" * 64)


def build_audio(**overrides) -> AudioSpec:
    fields = {
        "sample_rate": 32000,
        "window": 3.0,
        "hop": 3.0,
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


def test_audio_spec_geometry_carries_window_hop_and_pad():
    geometry = build_audio(window=3.0, hop=1.0, pad="drop").geometry

    assert geometry == WindowGeometry(window=3.0, hop=1.0, pad="drop")


def test_window_count_counts_whole_windows_at_a_hop():
    geometry = WindowGeometry(window=3.0, hop=3.0, pad="drop")

    assert window_count(30.0, geometry) == 10


def test_window_count_is_zero_for_audio_shorter_than_one_window_when_dropping():
    geometry = WindowGeometry(window=3.0, hop=3.0, pad="drop")

    assert window_count(1.5, geometry) == 0


@pytest.mark.parametrize("pad", ["centre_crop_end_pad", "time_scaled"])
def test_window_count_is_one_for_audio_shorter_than_one_window_when_padding(pad):
    assert window_count(1.5, WindowGeometry(window=3.0, hop=3.0, pad=pad)) == 1


@pytest.mark.parametrize(
    ("pad", "expected"),
    [("drop", 3), ("centre_crop_end_pad", 4), ("time_scaled", 4)],
)
def test_window_count_drops_a_trailing_partial_window_and_pads_it_under_the_other_policies(
    pad, expected
):
    assert window_count(10.0, WindowGeometry(window=3.0, hop=3.0, pad=pad)) == expected


# Each duration is a whole number of hops past the window on paper and a hair off it
# in binary floating point: three hops lands just under the boundary and seven just
# over, so a count without a tolerance is wrong on one side or the other.
@pytest.mark.parametrize(("hops", "expected"), [(3, 4), (7, 8)])
@pytest.mark.parametrize("pad", ["drop", "centre_crop_end_pad", "time_scaled"])
def test_window_count_is_stable_at_an_exact_window_boundary(pad, hops, expected):
    duration = 3.0 + sum([0.1] * hops)

    assert window_count(duration, WindowGeometry(window=3.0, hop=0.1, pad=pad)) == expected
