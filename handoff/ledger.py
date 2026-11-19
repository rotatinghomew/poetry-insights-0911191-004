"""Append-only decision ledger.

Every destination records the decision version, the reasons it was queued or
held back, and the data limitations that still require manual triage.  Later
reviews are for the *next* go-live decision; they never rewrite a record that
was already handed over.

The ledger is JSON Lines so it can be appended atomically one record at a
time and inspected with standard tools.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .queue import QueueDecision
from .schema import Dataset


class LedgerError(ValueError):
    pass


def _canonical(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_ledger(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise LedgerError(f"{path}:{number}: invalid JSON: {error}") from error
            if not isinstance(record, dict) or "decision_id" not in record:
                raise LedgerError(f"{path}:{number}: record has no decision_id")
            records.append(record)
    return records


def decision_id(decision: QueueDecision) -> str:
    """Stable identity of one handoff destination under one decision version.

    The version is part of the identity on purpose: a later review informs the
    next go-live decision and receives new records, while a re-run of the same
    version must not rewrite (or silently replace) what was already handed over.
    """

    payload = {
        "queue_date": decision.queue_date,
        "family_id": decision.family_id,
        "team": decision.team,
        "labels": decision.labels,
        "decision_version": decision.decision_version,
    }
    from .schema import stable_id

    parts = [f"{key}={_canonical(value)}" for key, value in sorted(payload.items())]
    return stable_id("Q", parts)


def append_decisions(
    path: str | Path,
    decisions: Iterable[QueueDecision],
    *,
    dataset: Dataset | None = None,
    note: str = "",
    now: datetime | None = None,
) -> list[dict]:
    """Append decisions, refusing to change an already recorded decision.

    A decision with the same id is only accepted if its payload is identical.
    Re-running a later tool version produces different decision ids, so old
    hand-offs stay intact while the new run is recorded separately.
    """

    path = Path(path)
    existing = read_ledger(path)
    by_id = {record["decision_id"]: record for record in existing}
    timestamp = (now or datetime.now(timezone.utc)).isoformat()
    added: list[dict] = []
    conflicts: list[str] = []

    for decision in decisions:
        identifier = decision_id(decision)
        record = {
            "decision_id": identifier,
            "recorded_at": timestamp,
            "queue_date": decision.queue_date,
            "family_id": decision.family_id,
            "representative_id": decision.representative_id,
            "message_ids": list(decision.message_ids),
            "team": decision.team,
            "shift": decision.shift,
            "labels": list(decision.labels),
            "label_sources": dict(decision.label_sources),
            "action": decision.action,
            "priority": decision.priority,
            "mandatory": decision.mandatory,
            "impact": decision.impact,
            "novel": decision.novel,
            "channel": decision.channel,
            "submitted_at": decision.submitted_at,
            "reasons": list(decision.reasons),
            "limitations": list(decision.limitations),
            "scores": dict(decision.scores),
            "decision_version": decision.decision_version,
            "model_version": decision.model_version,
            "policy_version": decision.policy_version,
            "remaining_before": decision.remaining_before,
            "remaining_after": decision.remaining_after,
            "note": note,
        }
        if dataset is not None:
            record["inputs_digest"] = _inputs_digest(dataset)
        current = by_id.get(identifier)
        if current is not None:
            comparison = {key: value for key, value in record.items() if key != "recorded_at"}
            stored = {key: value for key, value in current.items() if key != "recorded_at"}
            if comparison != stored:
                conflicts.append(identifier)
                continue
            continue
        added.append(record)
        by_id[identifier] = record

    if conflicts:
        raise LedgerError(
            "refusing to rewrite handed-over decisions: " + ", ".join(sorted(conflicts))
        )

    if added:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as sink:
            for record in added:
                sink.write(_canonical(record) + "\n")
    return added


def _inputs_digest(dataset: Dataset) -> str:
    from .schema import snapshot_digest

    return snapshot_digest(dataset)


def lookup(path: str | Path, decision_id_value: str) -> dict | None:
    for record in read_ledger(path):
        if record["decision_id"] == decision_id_value:
            return record
    return None


def lookup_message(path: str | Path, message_id: str) -> list[dict]:
    """Return every destination ever recorded for one message."""

    result = []
    for record in read_ledger(path):
        if message_id in record.get("message_ids", []):
            result.append(record)
    return result
