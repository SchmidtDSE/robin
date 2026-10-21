"""What the engine refuses to attempt, checked before any recording's input is fetched.

Nothing here loads, reads, opens or fetches anything: every argument is an
already-verified in-memory value. The checks on the card, the pinned files, the input,
the settings and the output requests are shared with the control plane. The two here
need the loaded registry or the work's resources.
"""

from robin_contracts.registry import TaxonRegistry
from robin_contracts.request_checks import check_request_for_problem
from robin_contracts.work import REGISTRY_ROLE, InferenceWork
from robin_inference_engine import errors
from robin_inference_engine.detections import DUCKDB_MEMORY_LIMIT


def refuse_request(work: InferenceWork, *, registry: TaxonRegistry | None) -> None:
    """Everything the engine refuses before it builds the model."""
    problem = check_request_for_problem(work.model, work.input, work.settings, work.outputs)
    if problem is not None:
        raise _refused(problem.code, problem.detail)
    _refuse_a_substituted_registry(work, registry)
    _refuse_a_duckdb_memory_limit_that_is_not_a_string(work)


def _refused(code: str, detail: str) -> errors.EngineError:
    return errors.EngineError(code, errors.VALIDATE_REQUEST, detail)


def _refuse_a_substituted_registry(
    work: InferenceWork, registry: TaxonRegistry | None
) -> None:
    # Compared with the bytes that were loaded, not only the file that was verified,
    # so a file replaced between the two reads is still refused.
    pinned = work.model.files.get(REGISTRY_ROLE)
    if pinned is None:
        return
    if registry is None:
        raise _refused(
            errors.REGISTRY_FINGERPRINT_MISMATCH,
            f"the work pins {REGISTRY_ROLE} file digest {pinned.digest} but no registry "
            f"was loaded",
        )
    if registry.fingerprint != pinned.digest:
        raise _refused(
            errors.REGISTRY_FINGERPRINT_MISMATCH,
            f"the work pins {REGISTRY_ROLE} file digest {pinned.digest} but the loaded "
            f"registry's fingerprint is {registry.fingerprint}",
        )


def _refuse_a_duckdb_memory_limit_that_is_not_a_string(work: InferenceWork) -> None:
    # DuckDB parses the value itself, when it builds the detections.
    limit = work.resources.get(DUCKDB_MEMORY_LIMIT)
    if DUCKDB_MEMORY_LIMIT in work.resources and not (isinstance(limit, str) and limit):
        raise _refused(
            errors.RESOURCE_INVALID,
            f"resource {DUCKDB_MEMORY_LIMIT!r} is {limit!r}, but DuckDB needs a non-empty "
            f"string such as '4GB'",
        )
