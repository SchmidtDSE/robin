import pytest
from pydantic import ValidationError

from robin_contracts.ports.compute import ComputeRequest


@pytest.mark.parametrize("field", ["requests", "limits"])
@pytest.mark.parametrize("entry", [{"": "2"}, {"cpu": ""}], ids=["empty-name", "empty-value"])
def test_compute_request_refuses_an_empty_name_or_value(field, entry):
    with pytest.raises(ValidationError):
        ComputeRequest(**{field: entry})


@pytest.mark.parametrize("deadline", [0, -1, True])
def test_compute_request_refuses_a_deadline_below_one(deadline):
    with pytest.raises(ValidationError):
        ComputeRequest(active_deadline_seconds=deadline)


def test_compute_request_round_trips_through_json():
    request = ComputeRequest(
        requests={"cpu": "2", "memory": "8Gi"},
        limits={"memory": "8Gi"},
        active_deadline_seconds=21600,
    )

    assert ComputeRequest.model_validate_json(request.model_dump_json()) == request
