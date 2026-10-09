from typing import get_args

import pytest

from robin_contracts.results import (
    RECORDING_FAILURE_STAGES,
    WORK_FAILURE_STAGES,
    FailureReport,
    InferenceFailure,
    RecordingFailed,
)
from robin_contracts.work import RecordingRef
from robin_inference_engine import errors

# The stages a failure may name, read from the contract a FailureReport validates
# against, so a stage the engine declares and the result refuses fails here.
DECLARED_STAGES = get_args(FailureReport.model_fields["stage"].annotation)

FAMILIES = (
    (errors.ACCEPT_WINDOW, errors.ACCEPT_WINDOW_FAILURES),
    (errors.VALIDATE_REQUEST, errors.VALIDATE_REQUEST_FAILURES),
    (errors.ACQUIRE_MODEL, errors.ACQUIRE_MODEL_FAILURES),
    (errors.LOAD_REGISTRY, errors.LOAD_REGISTRY_FAILURES),
    (errors.CONSTRUCT_MODEL, errors.CONSTRUCT_MODEL_FAILURES),
    (errors.ACQUIRE_INPUT, errors.ACQUIRE_INPUT_FAILURES),
    (errors.INFER, errors.INFER_FAILURES),
    (errors.READ_INPUT_ARTIFACT, errors.READ_INPUT_ARTIFACT_FAILURES),
    (errors.WRITE_ARTIFACT, errors.WRITE_ARTIFACT_FAILURES),
    (errors.AGGREGATE, errors.AGGREGATE_FAILURES),
)


def test_every_declared_constant_is_a_stage_or_one_of_its_codes():
    declared = {
        value
        for name, value in vars(errors).items()
        if name.isupper() and isinstance(value, str)
    }

    assert len(set(errors.ACCEPT_WINDOW_FAILURES)) == 14
    assert len(set(errors.VALIDATE_REQUEST_FAILURES)) == 16
    assert len(set(errors.ACQUIRE_MODEL_FAILURES)) == 2
    assert len(set(errors.LOAD_REGISTRY_FAILURES)) == 2
    assert len(set(errors.CONSTRUCT_MODEL_FAILURES)) == 5
    assert len(set(errors.ACQUIRE_INPUT_FAILURES)) == 1
    assert len(set(errors.INFER_FAILURES)) == 2
    assert len(set(errors.READ_INPUT_ARTIFACT_FAILURES)) == 10
    assert len(set(errors.WRITE_ARTIFACT_FAILURES)) == 3
    assert len(set(errors.AGGREGATE_FAILURES)) == 2
    assert declared == {stage for stage, _ in FAMILIES} | {
        code for _, codes in FAMILIES for code in codes
    }


def test_retention_and_setting_codes_belong_to_their_stages():
    assert errors.SCORE_BELOW_FLOOR == "score_below_floor"
    assert errors.SCORE_FLOOR_BELOW_MODEL_FLOOR == "score_floor_below_model_floor"
    assert errors.SETTING_UNDECLARED == "setting_undeclared"
    assert errors.SETTING_TYPE_MISMATCH == "setting_type_mismatch"
    assert errors.SCORE_BELOW_FLOOR in errors.ACCEPT_WINDOW_FAILURES
    assert {
        errors.SCORE_FLOOR_BELOW_MODEL_FLOOR,
        errors.SETTING_UNDECLARED,
        errors.SETTING_TYPE_MISMATCH,
    } <= set(errors.VALIDATE_REQUEST_FAILURES)


def test_no_code_belongs_to_two_stages():
    every = [code for _, codes in FAMILIES for code in codes]
    assert len(set(every)) == len(every)


def test_each_stage_constant_is_one_of_the_declared_stages():
    for stage, _ in FAMILIES:
        assert stage in DECLARED_STAGES


def report_at(stage: str, code: str) -> FailureReport:
    report = FailureReport(code=code, stage=stage, detail=f"{code} was raised")
    if stage in WORK_FAILURE_STAGES:
        return InferenceFailure(
            schema_version="robin.inference-result/1",
            work_digest="sha256:v1:" + "0" * 64,
            failure=report,
        ).failure
    return RecordingFailed(namespace="soundhub", value="42", failure=report).failure


@pytest.mark.parametrize(("stage", "codes"), FAMILIES)
def test_every_declared_code_is_reportable_at_its_stages_scope(stage, codes):
    for code in codes:
        assert (report_at(stage, code).code, report_at(stage, code).stage) == (code, stage)


def test_a_local_write_failure_fails_one_recording():
    assert errors.ARTIFACT_WRITE_FAILED == "artifact_write_failed"
    assert errors.ARTIFACT_WRITE_FAILED in errors.WRITE_ARTIFACT_FAILURES
    assert errors.WRITE_ARTIFACT in RECORDING_FAILURE_STAGES


RECORDING = RecordingRef(namespace="soundhub", value="42", audio_uri="s3://b/42.wav")


def test_engine_error_carries_its_code_and_location():
    error = errors.EngineError(
        errors.DUPLICATE_WINDOW,
        errors.ACCEPT_WINDOW,
        "recording ('soundhub', '42') already produced a window at 6.0",
        recording=RECORDING,
        window_start_s=6.0,
    )

    assert error.code == errors.DUPLICATE_WINDOW
    assert error.stage == "accept_window"
    assert error.detail == "recording ('soundhub', '42') already produced a window at 6.0"
    assert error.recording is RECORDING
    assert error.window_start_s == 6.0
    assert str(error) == error.detail
