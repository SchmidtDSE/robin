"""Allowed-edges table for ROBIN's inter-package import boundaries.

Plain-Python data, imported by ``test_import_edges.py`` and by the AST
checker helper defined there.
"""

FORBIDDEN_PREFIXES = ("soundhub_",)

# The robin_* imports that each package may make. The list is exhaustive.
FIXED_ALLOWED_ROBIN_IMPORTS = {
    "robin_contracts": frozenset(),
    "robin_inference_engine": frozenset({"robin_contracts"}),
    "robin_run_manager": frozenset({"robin_contracts", "robin_models"}),
    "robin_cli": frozenset({"robin_run_manager"}),
    "robin_adapters": frozenset({"robin_contracts"}),
    "robin_models": frozenset({"robin_contracts"}),
    "robin_worker": frozenset(
        {"robin_contracts", "robin_inference_engine", "robin_adapters", "robin_models"}
    ),
}

# The composition root is the one run-manager module that may import adapters.
COMPOSITION_ROOT = "robin_run_manager.compose"
COMPOSITION_ROOT_ALLOWED = frozenset({"robin_contracts", "robin_adapters"})
