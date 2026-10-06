"""Where a published artifact sits below a writer's root.

Every writer and every reader of the published data depends on this layout, so it is
built in one place. Readers outside Python decode the values by the same rule.
"""

from urllib.parse import quote

from robin_contracts.results import ArtifactKind

NAMESPACE_KEY = "recording_namespace"
VALUE_KEY = "recording_value"

FILE_NAMES: dict[ArtifactKind, str] = {
    "scores": "scores.parquet",
    "embeddings": "embeddings.parquet",
    "detections": "detections.parquet",
}


def artifact_path(kind: ArtifactKind, namespace: str, value: str) -> str:
    """The relative POSIX path of one recording's file of `kind`.

    Each value is percent-encoded outside `A-Z a-z 0-9 - . _ ~`, so a value holding
    `/` or `=` stays inside its own path segment.
    """
    return "/".join(
        [
            kind,
            f"{NAMESPACE_KEY}={quote(namespace, safe='')}",
            f"{VALUE_KEY}={quote(value, safe='')}",
            FILE_NAMES[kind],
        ]
    )
