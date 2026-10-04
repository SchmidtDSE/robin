import dataclasses
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import robin_contracts.inputs
from robin_contracts.canonical import canonical_json_bytes
from robin_contracts.inputs import AudioClip, Embeddings, Input


def build_embeddings() -> Embeddings:
    return Embeddings(
        starts=np.array([0.0, 3.0]),
        ends=np.array([3.0, 6.0]),
        values=np.zeros((2, 4), dtype=np.float32),
    )


def test_audio_clip_construction():
    clip = AudioClip(path=Path("/data/rec.wav"))
    assert clip.kind == "audio"
    assert clip.path == Path("/data/rec.wav")


def test_audio_clip_frozen():
    clip = AudioClip(path=Path("/data/rec.wav"))
    with pytest.raises(ValidationError):
        clip.path = Path("/data/other.wav")


def test_embeddings_hold_the_arrays_they_were_given():
    starts = np.array([0.0, 3.0])
    ends = np.array([3.0, 6.0])
    values = np.zeros((2, 4), dtype=np.float32)

    embeddings = Embeddings(starts=starts, ends=ends, values=values)

    assert embeddings.kind == "embeddings"
    assert embeddings.starts is starts
    assert embeddings.ends is ends
    assert embeddings.values is values


def test_embeddings_are_frozen():
    embeddings = build_embeddings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        embeddings.starts = np.array([1.0, 4.0])


def test_embeddings_are_built_by_keyword_only():
    with pytest.raises(TypeError):
        Embeddings("embeddings", np.array([0.0]), np.array([3.0]), np.zeros((1, 4)))


def test_inputs_does_not_bind_numpy_at_runtime():
    assert not hasattr(robin_contracts.inputs, "np")


def test_input_union_audio():
    clip = AudioClip(path=Path("/data/rec.wav"))
    assert isinstance(clip, AudioClip)
    inp: Input = clip
    assert inp.kind == "audio"


def test_input_union_embeddings():
    embeddings = build_embeddings()
    assert isinstance(embeddings, Embeddings)
    inp: Input = embeddings
    assert inp.kind == "embeddings"


def test_audio_clip_rejects_extra_fields():
    with pytest.raises(ValidationError):
        AudioClip(path=Path("/data/rec.wav"), extra="nope")


def test_audio_clip_canonical_wire_format():
    clip = AudioClip(path=Path("/data/rec.wav"))

    assert canonical_json_bytes(clip) == (
        b'{"kind":"audio","path":"/data/rec.wav"}'
    )
