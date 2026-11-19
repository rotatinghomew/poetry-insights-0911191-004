"""Command line interface for the handoff tool.

Examples::

    python3 -m handoff assess --data data
    python3 -m handoff queue --data data --ledger out/decisions.jsonl
    python3 -m handoff explain --ledger out/decisions.jsonl --message M05
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

from .ledger import LedgerError, append_decisions, decision_id, lookup_message, read_ledger
from .model import Assessment, TOOL_VERSION
from .queue import QueueBuilder
from .schema import DataError, load_dataset

QUEUE_COLUMNS = [
    "queue_date",
    "decision_id",
    "priority",
    "action",
    "team",
    "shift",
    "family_id",
    "representative_id",
    "message_ids",
    "labels",
    "label_sources",
    "mandatory",
    "impact",
    "novel",
    "channel",
    "submitted_at",
    "reasons",
    "limitations",
    "scores",
    "remaining_before",
    "remaining_after",
    "decision_version",
    "model_version",
    "policy_version",
]


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid date: {value}") from error


def _write_json(path: str | None, payload) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _write_csv(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=QUEUE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _decision_records(builder: QueueBuilder) -> list[dict]:
    records = []
    for decision in builder.decisions:
        record = decision.to_dict()
        record["decision_id"] = decision_id(decision)
        records.append(record)
    return records


def _queue_csv_rows(builder: QueueBuilder) -> list[dict]:
    rows = []
    for decision, record in zip(builder.decisions, _decision_records(builder)):
        row = decision.csv_row()
        row["decision_id"] = record["decision_id"]
        rows.append(row)
    return rows


def cmd_assess(args) -> int:
    dataset = load_dataset(args.data)
    assessment = Assessment(dataset, as_of=_parse_date(args.as_of))
    if args.label:
        try:
            payload = {
                "as_of": assessment.as_of.isoformat(),
                "decision_version": assessment.decision_version,
                **assessment.label_report(args.label),
            }
        except KeyError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    else:
        payload = assessment.summary()
    _write_json(args.out, payload)
    return 0


def cmd_queue(args) -> int:
    dataset = load_dataset(args.data)
    assessment = Assessment(dataset, as_of=_parse_date(args.as_of))
    builder = QueueBuilder(
        dataset,
        assessment,
        queue_date=_parse_date(args.queue_date),
        include_backlog=args.include_backlog,
    )
    builder.build()
    records = _decision_records(builder)
    payload = builder.summary()
    payload["decisions"] = records
    _write_json(args.out, payload)
    if args.csv:
        _write_csv(args.csv, _queue_csv_rows(builder))
    if args.ledger:
        added = append_decisions(args.ledger, builder.decisions, dataset=dataset, note=args.note)
        print(f"recorded {len(added)} new decision(s) in {args.ledger}", file=sys.stderr)
    if args.strict and any(
        decision.action == "overflow_mandatory" for decision in builder.decisions
    ):
        print("error: a mandatory item exceeds shift capacity", file=sys.stderr)
        return 3
    return 0


def cmd_explain(args) -> int:
    if args.ledger:
        records = lookup_message(args.ledger, args.message)
        if not records:
            print(f"no ledger record for {args.message}", file=sys.stderr)
            return 2
        _write_json(args.out, records)
        return 0
    dataset = load_dataset(args.data)
    assessment = Assessment(dataset, as_of=_parse_date(args.as_of))
    message = dataset.message(args.message)
    if message is None:
        print(f"unknown message: {args.message}", file=sys.stderr)
        return 2
    family = assessment.membership[message.message_id]
    novelty = assessment.novelty.get(message.message_id)
    payload = {
        "message": {
            "message_id": message.message_id,
            "channel": message.channel,
            "event_id": message.event_id,
            "submitted_at": message.submitted_at.raw,
            "text": message.text,
        },
        "family": {
            "family_id": family.family_id,
            "message_ids": family.message_ids,
            "links": family.links,
        },
        "novelty": {
            "novel": bool(novelty and novelty.novel),
            "segments": list(novelty.segments) if novelty else [],
            "reason": novelty.reason if novelty else "",
        },
        "scores": {
            label: assessment.score(label, message.text)
            for label in assessment.label_catalog
        },
        "labels": {
            label: assessment.label_report(label) for label in assessment.label_catalog
        },
        "decision_version": assessment.decision_version,
    }
    _write_json(args.out, payload)
    return 0


def cmd_ledger(args) -> int:
    records = read_ledger(args.path)
    if args.message:
        records = [record for record in records if args.message in record.get("message_ids", [])]
    _write_json(args.out, records)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="handoff",
        description="Per-label go-live assessment and auditable review handoff queue.",
    )
    parser.add_argument("--version", action="version", version=TOOL_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_data_options(sub):
        sub.add_argument("--data", default="data", help="directory containing the CSV exports")
        sub.add_argument("--as-of", help="planning cut-off date (YYYY-MM-DD)")
        return sub

    assess = add_data_options(subparsers.add_parser("assess", help="per-label go-live assessment"))
    assess.add_argument("--label", help="report one label only")
    assess.add_argument("--out", help="write JSON to this path instead of stdout")
    assess.set_defaults(func=cmd_assess)

    queue = add_data_options(subparsers.add_parser("queue", help="build the next-day handoff queue"))
    queue.add_argument("--queue-date", help="queue date (default: day after the cut-off)")
    queue.add_argument("--include-backlog", action="store_true", help="include earlier messages")
    queue.add_argument("--out", help="write JSON summary to this path")
    queue.add_argument("--csv", help="write a CSV queue for the duty rota")
    queue.add_argument("--ledger", help="append the decisions to this JSONL ledger")
    queue.add_argument("--note", default="", help="free-text note stored with new ledger records")
    queue.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if a mandatory item overflows capacity",
    )
    queue.set_defaults(func=cmd_queue)

    explain = subparsers.add_parser("explain", help="explain one message or ledger record")
    explain.add_argument("message", help="message id")
    explain.add_argument("--data", default="data", help="directory containing the CSV exports")
    explain.add_argument("--as-of", help="planning cut-off date (YYYY-MM-DD)")
    explain.add_argument("--ledger", help="read a decision ledger instead of fresh inputs")
    explain.add_argument("--out", help="write JSON to this path instead of stdout")
    explain.set_defaults(func=cmd_explain)

    ledger = subparsers.add_parser("ledger", help="inspect a decision ledger")
    ledger.add_argument("path", help="JSONL ledger path")
    ledger.add_argument("--message", help="filter by message id")
    ledger.add_argument("--out", help="write JSON to this path instead of stdout")
    ledger.set_defaults(func=cmd_ledger)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (DataError, LedgerError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
