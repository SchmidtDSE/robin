"""The composition root: it builds what the operations use from a profile.

This is the one module in `robin_run_manager` that may import adapters. It imports
only the adapters that the profile names.
"""

from dataclasses import dataclass

from robin_run_manager.database import Database, open_database
from robin_run_manager.profile import Profile


@dataclass(frozen=True)
class Services:
    profile: Profile
    database: Database


def compose(profile: Profile) -> Services:
    return Services(profile=profile, database=open_database(profile.database))
