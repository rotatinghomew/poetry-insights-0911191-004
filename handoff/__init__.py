"""Per-label go-live assessment and auditable review handoff.

The package is standard-library only.  See :mod:`handoff.model`,
:mod:`handoff.queue` and :mod:`handoff.ledger` for the three decision layers.
"""

from .model import Assessment, Policy, TOOL_VERSION
from .queue import QueueBuilder
from .schema import DataError, Dataset, load_dataset

__all__ = [
    "Assessment",
    "DataError",
    "Dataset",
    "Policy",
    "QueueBuilder",
    "TOOL_VERSION",
    "load_dataset",
]
