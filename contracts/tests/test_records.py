import dataclasses

import pytest

import robin_contracts.records
from robin_contracts.records import WindowOutput


def test_window_output_defaults_to_no_scores_and_no_embedding():
    window = WindowOutput(start=0.0, end=3.0)

    assert window.scores is None
    assert window.embedding is None


def test_window_output_fields():
    names = {field.name for field in dataclasses.fields(WindowOutput)}

    assert names == {"start", "end", "scores", "embedding"}


def test_window_output_fields_are_keyword_only():
    # A call written against an older positional order must fail, not shift arguments.
    with pytest.raises(TypeError):
        WindowOutput(0, 0.0, 3.0)


def test_records_does_not_bind_numpy_at_runtime():
    assert not hasattr(robin_contracts.records, "np")
