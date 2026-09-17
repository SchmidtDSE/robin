from robin_inference_engine import errors


def test_acceptance_codes_are_unique_and_complete():
    declared = {
        value
        for name, value in vars(errors).items()
        if name.isupper() and isinstance(value, str)
    }

    assert len(set(errors.ACCEPTANCE_CODES)) == 13
    assert declared == set(errors.ACCEPTANCE_CODES) | {errors.ACCEPT_WINDOW}


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
