"""Allowed-edges table for ROBIN's inter-package import boundaries.

Plain-Python data, imported by ``test_import_edges.py`` and by the AST
checker helper defined there.
"""

FORBIDDEN_PREFIXES = ("soundhub_",)

# Packages whose robin_* imports are a fixed, exhaustive allow-list. A
# module whose top-level name matches none of these keys (currently only
# robin_worker) has no robin_*-import constraint beyond FORBIDDEN_PREFIXES.
FIXED_ALLOWED_ROBIN_IMPORTS = {
    "robin_contracts": frozenset(),
    "robin_inference_engine": frozenset({"robin_contracts"}),
    "robin_run_manager": frozenset({"robin_contracts"}),
}

# robin_adapters.* is governed separately because its persistence subtree
# carries an additional allowance that the rest of the adapters tree does
# not get.
ADAPTERS_PREFIX = "robin_adapters"
ADAPTERS_PERSISTENCE_PREFIX = "robin_adapters.persistence"
ADAPTERS_DEFAULT_ALLOWED = frozenset({"robin_contracts"})
ADAPTERS_PERSISTENCE_ALLOWED = frozenset({"robin_contracts", "robin_run_manager"})
