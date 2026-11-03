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
    "robin_cli": frozenset({"robin_run_manager", "robin_contracts"}),
    "robin_adapters": frozenset({"robin_contracts"}),
    "robin_models": frozenset({"robin_contracts"}),
    "robin_worker": frozenset(
        {"robin_contracts", "robin_inference_engine", "robin_adapters", "robin_models"}
    ),
}

# The composition root is the one run-manager module that may import adapters.
COMPOSITION_ROOT = "robin_run_manager.compose"
COMPOSITION_ROOT_ALLOWED = frozenset({"robin_contracts", "robin_adapters"})

# The one other run-manager module that may import from adapters: it joins the setup
# class of each compute kind, and may import only each kind's settings module.
COMPUTE_SETUPS = "robin_run_manager.compute_setups"
COMPUTE_SETUPS_ALLOWED = frozenset({"robin_contracts"})
COMPUTE_SETTINGS_MODULE = r"robin_adapters\.compute\.[a-z0-9_]+\.settings"

# The same for storage: it joins the settings class of each storage backend.
STORAGE_SETTINGS = "robin_run_manager.storage_settings"
STORAGE_SETTINGS_ALLOWED = frozenset({"robin_contracts"})
STORAGE_SETTINGS_MODULE = r"robin_adapters\.storage\.[a-z0-9_]+\.settings"
