from robin_inference_engine import errors


# Every stage a failure may name, spelled out so a renamed constant fails here.
DECLARED_STAGES = (
    "validate_request",
    "resolve_model",
    "load_registry",
    "acquire_audio",
    "read_input_artifact",
    "construct_model",
    "infer",
    "accept_window",
    "write_artifact",
    "aggregate",
)


def test_every_declared_constant_is_a_stage_or_one_of_its_codes():
    declared = {
        value
        for name, value in vars(errors).items()
        if name.isupper() and isinstance(value, str)
    }

    assert len(set(errors.ACCEPT_WINDOW_FAILURES)) == 13
    assert len(set(errors.VALIDATE_REQUEST_FAILURES)) == 10
    assert declared == (
        set(errors.ACCEPT_WINDOW_FAILURES)
        | set(errors.VALIDATE_REQUEST_FAILURES)
        | {errors.ACCEPT_WINDOW, errors.VALIDATE_REQUEST}
    )


def test_no_code_belongs_to_two_stages():
    assert not set(errors.ACCEPT_WINDOW_FAILURES) & set(errors.VALIDATE_REQUEST_FAILURES)

    both = errors.ACCEPT_WINDOW_FAILURES + errors.VALIDATE_REQUEST_FAILURES
    assert len(set(both)) == len(both)


def test_each_stage_constant_is_one_of_the_declared_stages():
    assert errors.ACCEPT_WINDOW in DECLARED_STAGES
    assert errors.VALIDATE_REQUEST in DECLARED_STAGES


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
