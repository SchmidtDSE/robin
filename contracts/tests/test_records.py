import dataclasses

import robin_contracts.records
from robin_contracts.records import ClassScore, WindowOutput


def test_class_score_accepts_a_value_the_boundary_will_refuse():
    score = ClassScore(label="", score=2.0)

    assert score.label == ""
    assert score.score == 2.0


def test_window_output_defaults_to_no_scores_and_no_embedding():
    window = WindowOutput(recording_index=0, start=0.0, end=3.0)

    assert window.scores == ()
    assert window.embedding is None


def test_window_output_has_no_recipe_field():
    names = {field.name for field in dataclasses.fields(WindowOutput)}

    assert names == {"recording_index", "start", "end", "scores", "embedding"}


def test_records_does_not_bind_numpy_at_runtime():
    assert not hasattr(robin_contracts.records, "np")
