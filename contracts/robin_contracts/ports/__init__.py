"""The ports: the services a caller passes to `run_work` for the engine to use.

A port is a protocol. The engine calls its methods, and an implementation satisfies it
by having the same methods, without importing it. The engine and robin's adapters may
not import each other, so the protocols are defined here, in the package both import.
Each port has its own module, such as `robin_contracts.ports.storage`.

Robin's own implementations live in `robin_adapters`, in one subpackage for each port
and one module for each backend, such as `robin_adapters.storage.local`.
A new port follows the same pattern.
"""
