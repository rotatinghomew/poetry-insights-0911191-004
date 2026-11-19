"""CSV loading and normalisation for the handoff tool.

The three input exports are deliberately small, flat CSV files.  Empty fields
mean "the export did not provide this"; they are never interpreted as false or
zero.  Optional columns are supported so that richer exports can be used
without changing the required schema.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Iterable, Sequence

TRUTHY = {"1", "true", "yes", "y", "是", "需要", "必复核", "must", "high", "高"}

_LABEL_SPLIT = re.compile(r"[;；|、,，]+")
_FORWARD_PREFIX = re.compile(
    r"^\s*(?:[\[【(（]\s*(?:转发|转帖|转)\s*[\]】)）]\s*[：:]?\s*"
    r"|(?:转发|转帖)\s*[：:]\s*"
    r"|FWD?\s*[：:]\s*)",
    re.IGNORECASE,
)
_FORWARD_SUFFIX = re.compile(
    r"\s*[\[【(（]\s*(?:转发|转帖|转)\s*[\]】)）]\s*$", re.IGNORECASE
)
_FORWARD_INLINE = re.compile(
    r"[\[【(（]\s*(?:转发|转帖|转)\s*[\]】)）]|(?:转发|转帖)\s*[：:]", re.IGNORECASE
)
_QUOTE_CHARS = "「」『』“”\"'‘’《》<>"
_PUNCT = re.compile(r"[\s,.!?;:，。！？；：、…—\-_/\\|()（）\[\]【】]+")
_LEAD_INS = (
    "我想反映一下",
    "我想反馈一下",
    "我想咨询一下",
    "我想问一下",
    "我想请问一下",
    "我想问问",
    "我想问",
    "我想反映",
    "我想反馈",
    "我想咨询",
    "想反映一下",
    "想反馈一下",
    "想请问一下",
    "想请教一下",
    "想问一下",
    "想请问",
    "想问",
    "想反映",
    "想反馈",
    "麻烦问一下",
    "麻烦问",
    "请问一下",
    "请问",
    "咨询一下",
    "反映一下",
    "反馈一下",
    "建议",
    "希望",
    "能否",
    "可不可以",
    "可以",
)
_TRAILERS = ("谢谢", "多谢", "感谢", "麻烦了", "辛苦", "多谢了", "谢谢你")


class DataError(ValueError):
    """Raised when an input export cannot be interpreted safely."""


@dataclass(frozen=True)
class Stamp:
    """A partial timestamp; exports may contain a date, a time, or both."""

    date: date | None = None
    time: time | None = None
    raw: str = ""

    @property
    def empty(self) -> bool:
        return self.date is None and self.time is None

    def with_default_date(self, fallback: date | None) -> "Stamp":
        if self.date is not None or fallback is None:
            return self
        return Stamp(fallback, self.time, self.raw)

    def sort_key(self, fallback: date | None = None) -> tuple:
        value = self.with_default_date(fallback)
        return (
            value.date or date.min,
            value.time or time.min,
        )

    def at_or_before(self, cutoff: "Stamp", fallback: date | None = None) -> bool:
        return self.sort_key(fallback) <= cutoff.sort_key(fallback)


def parse_stamp(value: str, *, field_name: str = "timestamp") -> Stamp:
    raw = (value or "").strip()
    if not raw:
        return Stamp()
    candidates = [raw, raw.replace("Z", "+00:00")]
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            return Stamp(parsed.date(), parsed.time().replace(tzinfo=None), raw)
        except ValueError:
            pass
        try:
            return Stamp(date.fromisoformat(candidate), None, raw)
        except ValueError:
            pass
        try:
            return Stamp(None, time.fromisoformat(candidate), raw)
        except ValueError:
            pass
    for fmt in (
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y%m%d %H:%M",
        "%Y年%m月%d日 %H:%M",
        "%Y年%m月%d日%H:%M",
    ):
        try:
            parsed = datetime.strptime(raw, fmt)
            return Stamp(parsed.date(), parsed.time(), raw)
        except ValueError:
            continue
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return Stamp(None, parsed.time(), raw)
        except ValueError:
            continue
    raise DataError(f"{field_name} cannot be parsed as a date/time: {raw!r}")


def split_labels(value: str) -> tuple[str, ...]:
    """Split a cell that may contain one or several reviewer labels."""

    if not value:
        return ()
    seen: list[str] = []
    for part in _LABEL_SPLIT.split(value):
        label = part.strip()
        if label and label not in seen:
            seen.append(label)
    return tuple(seen)


def truthy(value: str) -> bool:
    return (value or "").strip().lower() in TRUTHY


def canonical_text(text: str) -> str:
    """Normalise a message for duplicate/forward detection.

    Forward markers, quote wrappers, punctuation, whitespace and common
    politeness lead-ins are removed so that "改个开头再转发" and "引用同一段
    范文" collapse to the same fingerprint.  Meaning-bearing words are kept.
    """

    value = (text or "").strip()
    previous = None
    while value and value != previous:
        previous = value
        value = _FORWARD_PREFIX.sub("", value, count=1).strip()
        value = _FORWARD_SUFFIX.sub("", value, count=1).strip()
    value = _FORWARD_INLINE.sub("", value)
    value = value.strip(_QUOTE_CHARS)
    value = _PUNCT.sub("", value).lower()
    changed = True
    while changed and value:
        changed = False
        for lead in _LEAD_INS:
            if value.startswith(lead) and len(value) > len(lead):
                value = value[len(lead) :]
                changed = True
                break
    changed = True
    while changed and value:
        changed = False
        for trailer in _TRAILERS:
            if value.endswith(trailer) and len(value) > len(trailer):
                value = value[: -len(trailer)]
                changed = True
                break
    return value


def bigrams(value: str) -> set[str]:
    if len(value) < 2:
        return {value} if value else set()
    return {value[index : index + 2] for index in range(len(value) - 1)}


def text_similarity(left: str, right: str) -> float:
    left_grams = bigrams(left)
    right_grams = bigrams(right)
    if not left_grams or not right_grams:
        return 1.0 if left_grams == right_grams else 0.0
    shared = len(left_grams & right_grams)
    return shared / len(left_grams | right_grams)


def stable_id(prefix: str, parts: Iterable[str]) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


@dataclass(frozen=True)
class Message:
    message_id: str
    channel: str
    event_id: str
    submitted_at: Stamp
    text: str
    forwarded_from: tuple[str, ...] = ()
    impact: str = ""
    known_lead: bool = False
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def canonical(self) -> str:
        return canonical_text(self.text)


@dataclass(frozen=True)
class Review:
    review_id: str
    message_id: str
    reviewer: str
    labels: tuple[str, ...]
    closed_at: Stamp
    selection: str = "unknown"
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def closed(self) -> bool:
        # An empty closed_at means the record has not been settled.  Such a
        # record is never evidence of absence, nor a negative sample.
        return not self.closed_at.empty

    @property
    def selected_by_tool(self) -> bool:
        marker = (self.selection or "").strip().lower()
        return any(word in marker for word in ("tool", "model", "auto", "模型", "自动", "v0", "v1"))


@dataclass(frozen=True)
class Capacity:
    team: str
    shift: str
    capacity: int
    required_labels: tuple[str, ...]
    start_time: time | None = None
    end_time: time | None = None
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    def can_handle(self, label: str) -> bool:
        return label in self.required_labels


@dataclass
class Dataset:
    messages: list[Message]
    reviews: list[Review]
    capacities: list[Capacity]
    warnings: list[str] = field(default_factory=list)

    def message(self, message_id: str) -> Message | None:
        return self._messages_by_id.get(message_id)

    def __post_init__(self) -> None:
        self.messages.sort(key=lambda item: (item.submitted_at.sort_key(), item.message_id))
        self._messages_by_id = {item.message_id: item for item in self.messages}
        self._reviews_by_message: dict[str, list[Review]] = {}
        for review in self.reviews:
            self._reviews_by_message.setdefault(review.message_id, []).append(review)

    def reviews_for(self, message_id: str) -> list[Review]:
        return sorted(
            self._reviews_by_message.get(message_id, []),
            key=lambda item: (item.closed_at.sort_key(), item.review_id),
        )

    def closed_reviews(self) -> list[Review]:
        return [review for review in self.reviews if review.closed]

    def open_reviews(self) -> list[Review]:
        return [review for review in self.reviews if not review.closed]


def _read_rows(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise DataError(f"missing input file: {path}")
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise DataError(f"{path.name} has no header row")
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise DataError(f"{path.name} is missing columns: {', '.join(missing)}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            row["_line"] = str(line_number)
            rows.append(row)
    return rows


def _clean(row: dict[str, str], key: str) -> str:
    value = row.get(key)
    return (value or "").strip()


def load_dataset(data_dir: str | Path) -> Dataset:
    """Load and validate the three exports.

    Unknown message references and duplicate primary keys are hard errors: a
    hand-off that silently drops or merges records would be unauditable.
    """

    data_dir = Path(data_dir)
    warnings: list[str] = []

    message_rows = _read_rows(
        data_dir / "messages.csv",
        ("message_id", "channel", "event_id", "submitted_at", "text"),
    )
    messages: list[Message] = []
    seen_messages: set[str] = set()
    for row in message_rows:
        message_id = _clean(row, "message_id")
        if not message_id:
            raise DataError(f"messages.csv line {row['_line']}: empty message_id")
        if message_id in seen_messages:
            raise DataError(f"messages.csv: duplicate message_id {message_id}")
        seen_messages.add(message_id)
        channel = _clean(row, "channel")
        if not channel:
            raise DataError(f"messages.csv line {row['_line']}: empty channel for {message_id}")
        forwarded = split_labels(_clean(row, "forwarded_from"))
        messages.append(
            Message(
                message_id=message_id,
                channel=channel,
                event_id=_clean(row, "event_id"),
                submitted_at=parse_stamp(
                    _clean(row, "submitted_at"), field_name=f"messages.csv:{message_id}:submitted_at"
                ),
                text=_clean(row, "text"),
                forwarded_from=forwarded,
                impact=_clean(row, "impact"),
                known_lead=truthy(_clean(row, "known_lead")),
                raw=dict(row),
            )
        )

    review_rows = _read_rows(
        data_dir / "reviews.csv",
        ("review_id", "message_id", "reviewer", "label", "closed_at"),
    )
    reviews: list[Review] = []
    seen_reviews: set[str] = set()
    for row in review_rows:
        review_id = _clean(row, "review_id")
        message_id = _clean(row, "message_id")
        if not review_id:
            raise DataError(f"reviews.csv line {row['_line']}: empty review_id")
        if review_id in seen_reviews:
            raise DataError(f"reviews.csv: duplicate review_id {review_id}")
        seen_reviews.add(review_id)
        if message_id not in seen_messages:
            raise DataError(
                f"reviews.csv line {row['_line']}: unknown message_id {message_id!r} for {review_id}"
            )
        selection = (
            _clean(row, "selection")
            or _clean(row, "selected_by")
            or _clean(row, "review_batch")
        )
        if not selection:
            selection = "unknown"
        reviews.append(
            Review(
                review_id=review_id,
                message_id=message_id,
                reviewer=_clean(row, "reviewer"),
                labels=split_labels(_clean(row, "label")),
                closed_at=parse_stamp(
                    _clean(row, "closed_at"), field_name=f"reviews.csv:{review_id}:closed_at"
                ),
                selection=selection,
                raw=dict(row),
            )
        )

    capacity_rows = _read_rows(
        data_dir / "review_capacity.csv",
        ("team", "shift", "capacity", "required_labels"),
    )
    capacities: list[Capacity] = []
    for row in capacity_rows:
        team = _clean(row, "team")
        shift = _clean(row, "shift")
        if not team or not shift:
            raise DataError(f"review_capacity.csv line {row['_line']}: empty team or shift")
        try:
            capacity = int(_clean(row, "capacity"))
        except ValueError as error:
            raise DataError(
                f"review_capacity.csv line {row['_line']}: capacity must be an integer"
            ) from error
        if capacity < 0:
            raise DataError(f"review_capacity.csv line {row['_line']}: negative capacity")
        labels = split_labels(_clean(row, "required_labels"))
        if not labels:
            warnings.append(f"capacity row {team}/{shift} has no required_labels")
        start_time = _optional_time(_clean(row, "start_time") or _clean(row, "shift_start"))
        end_time = _optional_time(_clean(row, "end_time") or _clean(row, "shift_end"))
        capacities.append(
            Capacity(
                team=team,
                shift=shift,
                capacity=capacity,
                required_labels=labels,
                start_time=start_time,
                end_time=end_time,
                raw=dict(row),
            )
        )

    dataset = Dataset(messages=messages, reviews=reviews, capacities=capacities, warnings=warnings)
    dataset_as_of = infer_as_of(dataset)

    # A forward link to a missing message is a lineage gap.  Keep the link for
    # audit, but say so rather than pretending the family is complete.
    for message in messages:
        for parent in message.forwarded_from:
            if parent not in seen_messages:
                warnings.append(
                    f"{message.message_id} references missing forward source {parent}"
                )

    for review in reviews:
        if review.closed and not review.labels:
            warnings.append(f"review {review.review_id} is closed but carries no label")
        if review.closed:
            message = dataset.message(review.message_id)
            if message is not None and not message.submitted_at.empty:
                if review.closed_at.sort_key(dataset_as_of) < message.submitted_at.sort_key(dataset_as_of):
                    warnings.append(
                        f"review {review.review_id} was closed before its message was submitted"
                    )

    return dataset


def _optional_time(value: str) -> time | None:
    if not value:
        return None
    stamp = parse_stamp(value, field_name="shift time")
    return stamp.time


def infer_as_of(dataset: Dataset) -> date:
    dates = [message.submitted_at.date for message in dataset.messages if message.submitted_at.date]
    if dates:
        return max(dates)
    return date.today()


def snapshot_digest(dataset: Dataset, *, extra: str = "") -> str:
    """Stable digest of the input rows, used to version decisions."""

    parts = []
    for message in dataset.messages:
        parts.append("|".join([message.message_id, message.channel, message.event_id, message.submitted_at.raw, message.text]))
    for review in sorted(dataset.reviews, key=lambda item: item.review_id):
        parts.append(
            "|".join(
                [review.review_id, review.message_id, ",".join(review.labels), review.closed_at.raw, review.selection]
            )
        )
    for capacity in sorted(dataset.capacities, key=lambda item: (item.team, item.shift)):
        parts.append(
            "|".join(
                [capacity.team, capacity.shift, str(capacity.capacity), ",".join(capacity.required_labels)]
            )
        )
    parts.append(extra)
    return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()
