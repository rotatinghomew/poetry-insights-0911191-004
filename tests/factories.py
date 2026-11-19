"""Helpers for building temporary CSV exports in tests."""

from __future__ import annotations

import csv
from pathlib import Path

MESSAGE_COLUMNS = ["message_id", "channel", "event_id", "submitted_at", "text"]
EXTRA_MESSAGE_COLUMNS = ["forwarded_from", "impact", "known_lead"]
REVIEW_COLUMNS = ["review_id", "message_id", "reviewer", "label", "closed_at"]
EXTRA_REVIEW_COLUMNS = ["selection"]
CAPACITY_COLUMNS = ["team", "shift", "capacity", "required_labels"]


def write_dataset(
    directory: Path,
    *,
    messages: list[dict],
    reviews: list[dict],
    capacities: list[dict],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    _write(
        directory / "messages.csv",
        MESSAGE_COLUMNS + EXTRA_MESSAGE_COLUMNS,
        messages,
    )
    _write(directory / "reviews.csv", REVIEW_COLUMNS + EXTRA_REVIEW_COLUMNS, reviews)
    _write(directory / "review_capacity.csv", CAPACITY_COLUMNS, capacities)
    return directory


def _write(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def message(message_id, event_id, submitted_at, text, channel="线上", **extra):
    row = {
        "message_id": message_id,
        "channel": channel,
        "event_id": event_id,
        "submitted_at": submitted_at,
        "text": text,
    }
    row.update(extra)
    return row


def review(review_id, message_id, label, closed_at, *, reviewer="U1", selection="organic"):
    return {
        "review_id": review_id,
        "message_id": message_id,
        "reviewer": reviewer,
        "label": label,
        "closed_at": closed_at,
        "selection": selection,
    }


def capacity(team, shift, count, labels):
    return {
        "team": team,
        "shift": shift,
        "capacity": str(count),
        "required_labels": labels,
    }


def dense_dataset(directory: Path, *, extra_messages=None, extra_reviews=None, capacities=None) -> Path:
    """Three event batches with organic, settled, independent reviews.

    E1/E2 supply training and threshold data, E3 is the recent validation
    batch.  ``场地指引`` has enough 现场 evidence to pass; ``线上`` has too few
    positives.  ``教学建议`` has too few recent independent positives and must
    stay manual.  Extra rows let a test add "today's" messages.
    """

    messages = []
    reviews = []

    def add(message_id, event, day, index, text, channel, label):
        messages.append(
            message(message_id, event, f"{day} 10:{index:02d}", text, channel=channel)
        )
        reviews.append(review(f"V-{message_id}", message_id, label, f"{day} 12:{index:02d}"))

    layout = [
        ("场地指引", "无障碍坡道入口在哪里", "现场"),
        ("场地指引", "轮椅区域被临时围挡", "现场"),
        ("场地指引", "安检口指引不清晰", "现场"),
        ("场地指引", "志愿者指引牌放错了", "现场"),
        ("内容无障碍", "字幕字体太小", "现场"),
        ("内容无障碍", "台上朗诵听不清", "现场"),
        ("教学建议", "希望有适合低年级的互动", "现场"),
        ("场地指引", "停车区入口在哪里", "现场"),
        ("场地指引", "换场时通道怎么走", "线上"),
        ("内容无障碍", "朗诵能否增加字幕", "线上"),
        ("内容无障碍", "需要手语翻译", "线上"),
        ("教学建议", "能否提供教师朗读稿", "线上"),
    ]
    for event_index, event in enumerate(("E1", "E2", "E3"), start=1):
        day = f"2026-11-{10 + event_index:02d}"
        for item_index, (label, text, channel) in enumerate(layout, start=1):
            add(
                f"{event}-{item_index}",
                event,
                day,
                item_index,
                f"{text}（{event}第{item_index}条）",
                channel,
                label,
            )

    messages.extend(extra_messages or [])
    reviews.extend(extra_reviews or [])
    if capacities is None:
        capacities = [
            capacity("现场组", "早班", 5, "场地指引"),
            capacity("内容组", "早班", 5, "内容无障碍"),
            capacity("教研组", "早班", 5, "教学建议"),
        ]
    return write_dataset(directory, messages=messages, reviews=reviews, capacities=capacities)
