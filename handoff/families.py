"""Forward/duplicate families and novelty detection.

Two messages belong to one family when they are explicitly linked as forwards,
share an identical canonical body, or are near-identical within the same event.
A family is the unit of independent evidence: one review of a forward chain
must not be counted twice, and training/validation splits must keep families
whole.

A later member of a family may nevertheless add a substantive request.  Such a
message keeps its family link for audit, but is marked novel so that the old
conclusion cannot silently absorb it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schema import Dataset, Message, canonical_text, stable_id, text_similarity

EXACT_MATCH = 1.0
NEAR_DUPLICATE = 0.9
MIN_FUZZY_LENGTH = 4
MIN_CONTAINMENT_LENGTH = 6
MIN_CONTAINMENT_SHARE = 0.3

_TOKEN_SPLIT = re.compile(r"[，。！？；：、,.!?;:\s]+")


class _UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


@dataclass
class Family:
    family_id: str
    members: list[Message] = field(default_factory=list)
    links: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def representative(self) -> Message:
        return self.members[0]

    @property
    def message_ids(self) -> list[str]:
        return [member.message_id for member in self.members]


@dataclass(frozen=True)
class Novelty:
    novel: bool
    segments: tuple[str, ...] = ()
    reason: str = ""


def _contains(longer: str, shorter: str) -> bool:
    """A quoted body must be a real message, not a shared stock phrase.

    Two unrelated requests can share a short phrase ("增加字幕"); merging them
    would lose an independent evidence point.  A quoted body therefore has to
    be at least six canonical characters and make up a meaningful share of the
    longer text.  A genuine quote plus appended request clears both bars even
    though the appended part may be longer than the quote.
    """

    if len(shorter) < MIN_CONTAINMENT_LENGTH or shorter not in longer:
        return False
    return len(shorter) / len(longer) >= MIN_CONTAINMENT_SHARE


def _is_same_text_family(left: str, right: str) -> bool:
    """True when one body quotes the other with only appended new text."""

    if not left or not right:
        return False
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    if len(shorter) < MIN_FUZZY_LENGTH:
        return False
    if _contains(longer, shorter):
        return True
    return text_similarity(left, right) >= NEAR_DUPLICATE


def build_families(dataset: Dataset) -> dict[str, Family]:
    """Return families keyed by family_id, with members in submission order."""

    ordered = sorted(
        dataset.messages,
        key=lambda item: (item.submitted_at.sort_key(), item.message_id),
    )
    union = _UnionFind([message.message_id for message in ordered])
    links: list[tuple[str, str, str]] = []
    by_id = {message.message_id: message for message in ordered}

    for message in ordered:
        for parent in message.forwarded_from:
            if parent in by_id:
                union.union(message.message_id, parent)
                links.append((message.message_id, parent, "forward_link"))

    # Exact canonical bodies are the same evidence even across events/channels.
    canonical: dict[str, str] = {}
    for message in ordered:
        body = message.canonical
        if not body:
            continue
        if body in canonical:
            other = canonical[body]
            if union.find(message.message_id) != union.find(other):
                links.append((message.message_id, other, "identical_body"))
            union.union(message.message_id, other)
        else:
            canonical[body] = message.message_id

    # A quoted or lightly extended body is the same evidence.  Cross-event
    # matching requires containment, so a shared stock phrase alone does not
    # merge unrelated events.
    for index, message in enumerate(ordered):
        if not message.canonical:
            continue
        for other in ordered[:index]:
            if not other.canonical:
                continue
            if union.find(message.message_id) == union.find(other.message_id):
                continue
            if other.event_id != message.event_id and _contains(
                message.canonical, other.canonical
            ):
                union.union(message.message_id, other.message_id)
                links.append((message.message_id, other.message_id, "cross_event_quote"))
                continue
            if _is_same_text_family(message.canonical, other.canonical):
                union.union(message.message_id, other.message_id)
                links.append((message.message_id, other.message_id, "near_duplicate"))

    groups: dict[str, list[Message]] = {}
    for message in ordered:
        groups.setdefault(union.find(message.message_id), []).append(message)

    families: dict[str, Family] = {}
    by_root: dict[str, Family] = {}
    for root, members in groups.items():
        members.sort(key=lambda item: (item.submitted_at.sort_key(), item.message_id))
        family_id = stable_id("F", [member.message_id for member in members])
        family = Family(family_id=family_id, members=members)
        families[family_id] = family
        by_root[root] = family

    for child, parent, kind in links:
        family = by_root[union.find(child)]
        family.links.append((child, parent, kind))

    for family in families.values():
        family.links.sort()
    return families


def family_of(dataset: Dataset, families: dict[str, Family]) -> dict[str, Family]:
    mapping: dict[str, Family] = {}
    for family in families.values():
        for member in family.members:
            mapping[member.message_id] = family
    return mapping


def _segments(text: str) -> list[str]:
    stripped = canonical_text(text)
    if not stripped:
        return []
    result: list[str] = []
    for token in _TOKEN_SPLIT.split(text):
        normalized = canonical_text(token)
        if len(normalized) >= 4 and normalized not in result:
            result.append(normalized)
    if not result:
        result.append(stripped)
    return result


def _residual(segment: str, prior_texts: list[str]) -> tuple[bool, str]:
    """Return (covered, remainder) for one segment against earlier texts.

    If an earlier message is wholly contained in the segment, whatever is left
    over is new.  Otherwise fall back to near-duplicate similarity.
    """

    for prior in prior_texts:
        if segment in prior:
            # The new message adds nothing beyond the earlier one.
            return True, ""
        if prior and prior in segment:
            remainder = segment.replace(prior, "", 1)
            if len(remainder) >= 4:
                return False, remainder
            return True, ""
    if any(text_similarity(segment, prior) >= NEAR_DUPLICATE for prior in prior_texts):
        return True, ""
    return False, segment


def detect_novelty(message: Message, prior_members: list[Message]) -> Novelty:
    """Flag text added beyond the earlier members of the family.

    Politeness lead-ins and forward markers are ignored by ``canonical_text``.
    An earlier message that is wholly quoted leaves a remainder; a remainder
    of four or more characters is a substantive addition needing its own
    review.  So quoting a template verbatim is not new evidence, while an
    appended request is.
    """

    if not prior_members:
        return Novelty(False)
    prior_texts = [member.canonical for member in prior_members if member.canonical]
    if not prior_texts:
        return Novelty(False)

    additions: list[str] = []
    for segment in _segments(message.text):
        covered, remainder = _residual(segment, prior_texts)
        if covered:
            continue
        if remainder and remainder not in additions:
            additions.append(remainder)
    if additions:
        return Novelty(True, tuple(additions), "added_text")
    return Novelty(False)


def family_novelty(family: Family) -> dict[str, Novelty]:
    result: dict[str, Novelty] = {}
    for index, message in enumerate(family.members):
        result[message.message_id] = detect_novelty(message, family.members[:index])
    return result


def annotate_families(dataset: Dataset) -> tuple[dict[str, Family], dict[str, Family], dict[str, Novelty]]:
    families = build_families(dataset)
    membership = family_of(dataset, families)
    novelty: dict[str, Novelty] = {}
    for family in families.values():
        novelty.update(family_novelty(family))
    return families, membership, novelty
