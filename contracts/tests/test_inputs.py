from pathlib import Path

import pytest
from pydantic import ValidationError

from robin_contracts.inputs import AudioClip, Embedding, Input


def test_audio_clip_construction():
    clip = AudioClip(recording_id=1, path=Path("/data/rec.wav"))
    assert clip.kind == "audio"
    assert clip.recording_id == 1
    assert clip.path == Path("/data/rec.wav")


def test_audio_clip_frozen():
    clip = AudioClip(recording_id=1, path=Path("/data/rec.wav"))
    with pytest.raises(ValidationError):
        clip.recording_id = 2


def test_embedding_construction():
    emb = Embedding(recording_id=1, start=0.0, end=3.0, values=(0.1, 0.2, 0.3))
    assert emb.kind == "embedding"
    assert emb.values == (0.1, 0.2, 0.3)


def test_embedding_frozen():
    emb = Embedding(recording_id=1, start=0.0, end=3.0, values=(0.1,))
    with pytest.raises(ValidationError):
        emb.start = 1.0


def test_input_union_audio():
    clip = AudioClip(recording_id=1, path=Path("/data/rec.wav"))
    assert isinstance(clip, AudioClip)
    inp: Input = clip
    assert inp.kind == "audio"


def test_input_union_embedding():
    emb = Embedding(recording_id=1, start=0.0, end=3.0, values=(0.5,))
    assert isinstance(emb, Embedding)
    inp: Input = emb
    assert inp.kind == "embedding"


def test_audio_clip_rejects_extra_fields():
    with pytest.raises(ValidationError):
        AudioClip(recording_id=1, path=Path("/data/rec.wav"), extra="nope")


def test_embedding_rejects_extra_fields():
    with pytest.raises(ValidationError):
        Embedding(recording_id=1, start=0.0, end=3.0, values=(0.1,), extra="nope")
