from typing import get_args

import pytest

from robin_contracts.results import FailureReport
from robin_inference_engine import errors

# The stages a failure may name, read from the contract a FailureReport validates
# against, so a stage the engine declares and the result refuses fails here.
DECLARED_STAGES = get_args(FailureReport.model_fields["stage"].annotation)

FAMILIES = (
    (errors.ACCEPT_WINDOW, errors.ACCEPT_WINDOW_FAILURES),
    (errors.VALIDATE_REQUEST, errors.VALIDATE_REQUEST_FAILURES),
    (errors.INFER, errors.INFER_FAILURES),
)


def test_every_declared_constant_is_a_stage_or_one_of_its_codes():
    declared = {
        value
        for name, value in vars(errors).items()
        if name.isupper() and isinstance(value, str)
    }

    assert len(set(errors.ACCEPT_WINDOW_FAILURES)) == 13
    assert len(set(errors.VALIDATE_REQUEST_FAILURES)) == 10
    assert len(set(errors.INFER_FAILURES)) == 1
    assert declared == {stage for stage, _ in FAMILIES} | {
        code for _, codes in FAMILIES for code in codes
    }


def test_no_code_belongs_to_two_stages():
    every = [code for _, codes in FAMILIES for code in codes]
    assert len(set(every)) == len(every)


def test_each_stage_constant_is_one_of_the_declared_stages():
    for stage, _ in FAMILIES:
        assert stage in DECLARED_STAGES


@pytest.mark.parametrize(("stage", "codes"), FAMILIES)
def test_every_declared_code_is_reportable_on_its_own_stage(stage, codes):
    for code in codes:
        report = FailureReport(code=code, stage=stage, detail=f"{code} was raised")

        assert report.code == code
        assert report.stage == stage


def test_engine_error_carries_its_code_and_location():
    error = errors.EngineError(
        errors.DUPLICATE_WINDOW,
        errors.ACCEPT_WINDOW,
        "recording 2 already produced a window at 6.0",
        recording_index=2,
        window_start_s=6.0,
    )

    assert error.code == errors.DUPLICATE_WINDOW
    assert error.stage == "accept_window"
    assert error.detail == "recording 2 already produced a window at 6.0"
    assert error.recording_index == 2
    assert error.window_start_s == 6.0
    assert str(error) == error.detail
