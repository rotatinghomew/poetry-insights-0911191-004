"""Next-day handoff queue.

The queue is not a single confidence ranking.  Each message is split into
per-team requests, then ordered by explicit tiers:

0. mandatory known leads,
1. high-impact issues (especially ones that only appear on a small channel),
2. novel substantive appeals inside an otherwise duplicate family,
3. other items that still need a human.

Capacity is consumed per team and shift, so one saturated team cannot suppress
a request that another team is able to handle.  Duplicate/forward families
share a review, but a novel member gets its own seat so an old conclusion
cannot make it disappear.  Items the model is allowed to route do not consume
review seats; they still get an audit record.

Label evidence for the queue comes in three strengths, always recorded:

* ``review_closed``: a settled review of this forward family;
* ``review_open``: an unsettled review label, used only to find a candidate
  team, never as approval evidence;
* ``model`` / ``similar_closed``: the validated model or a settled neighbour;
* ``similar_open_provisional``: an unsettled neighbour, provisional only.

Provisional candidates are always marked manual and never auto-routed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, time, timedelta
from typing import Sequence

from .families import Family
from .model import Assessment, impact_rank
from .schema import Dataset, Message, Review, canonical_text

DEFAULT_SHIFT_WINDOWS = {
    "早班": (time(8, 0), time(14, 0)),
    "中班": (time(10, 0), time(18, 0)),
    "晚班": (time(14, 0), time(22, 0)),
    "夜班": (time(22, 0), time(6, 0)),
}

MANDATORY_LABELS = {"必复核", "mandatory"}

ACTION_QUEUED = "queued"
ACTION_QUEUED_MANUAL = "queued_manual"
ACTION_AUTO_ROUTED = "auto_routed"
ACTION_COVERED = "covered_by_family"
ACTION_ALREADY_OPEN = "already_open"
ACTION_DEFERRED = "deferred_no_capacity"
ACTION_OVERFLOW = "overflow_mandatory"
ACTION_NO_OWNER = "no_owner"

PRIORITY_MANDATORY = 0
PRIORITY_HIGH_IMPACT = 1
PRIORITY_NOVEL = 2
PRIORITY_NORMAL = 3
PRIORITY_AUDIT = 9

# Manual-only sources never auto-route.
MANUAL_ONLY_SOURCES = {"review_open", "similar_open_provisional", "manual_triage"}
PROVISIONAL_SOURCES = {"review_open", "similar_open_provisional"}
REVIEW_SOURCES = {"review_closed", "review_open"}


SIMILARITY_MIN_LENGTH = 6
SIMILARITY_MIN_SCORE = 0.6


def containment(left: str, right: str) -> float:
    """Share of the shorter text's bigrams found in the longer text.

    Very short texts are excluded: after politeness lead-ins are stripped, a
    request like "增加字幕" is only four characters, and every bigram of it can
    appear inside an unrelated longer sentence.  Treating that as evidence would
    let a settled neighbour lend a label it does not really support.
    """

    if min(len(left), len(right)) < SIMILARITY_MIN_LENGTH:
        return 0.0
    left_grams = {left[index : index + 2] for index in range(len(left) - 1)}
    right_grams = {right[index : index + 2] for index in range(len(right) - 1)}
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / min(len(left_grams), len(right_grams))


@dataclass
class ShiftSlot:
    team: str
    shift: str
    capacity: int
    start: time
    end: time
    used: int = 0
    preexisting: int = 0

    @property
    def remaining(self) -> int:
        return max(self.capacity - self.used, 0)

    def to_dict(self) -> dict:
        return {
            "team": self.team,
            "shift": self.shift,
            "capacity": self.capacity,
            "start": self.start.strftime("%H:%M"),
            "end": self.end.strftime("%H:%M"),
            "preexisting": self.preexisting,
            "assigned": self.used - self.preexisting,
            "remaining": self.remaining,
        }


@dataclass
class QueueDecision:
    queue_date: str
    message_ids: list[str]
    representative_id: str
    family_id: str
    team: str
    shift: str
    labels: list[str]
    action: str
    priority: int
    mandatory: bool
    impact: str
    novel: bool
    reasons: list[str]
    limitations: list[str]
    decision_version: str
    model_version: str
    policy_version: str
    channel: str
    submitted_at: str
    label_sources: dict[str, str] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    remaining_before: int | None = None
    remaining_after: int | None = None

    def priority_key(self) -> tuple:
        return (
            self.priority,
            -impact_rank(self.impact),
            0 if self.novel else 1,
            self.submitted_at,
            self.family_id,
            self.team,
            ";".join(self.labels),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def csv_row(self) -> dict:
        record = self.to_dict()
        record["message_ids"] = ";".join(self.message_ids)
        record["labels"] = ";".join(self.labels)
        record["reasons"] = ";".join(self.reasons)
        record["limitations"] = ";".join(self.limitations)
        record["label_sources"] = json.dumps(self.label_sources, ensure_ascii=False, sort_keys=True)
        record["scores"] = json.dumps(self.scores, ensure_ascii=False, sort_keys=True)
        return record


@dataclass
class Candidate:
    label: str
    source: str
    score: float


@dataclass
class FamilyContext:
    family: Family
    settled: list[Review]
    open: list[Review]
    mandatory: bool

    @property
    def closed_labels(self) -> set[str]:
        return {label for review in self.settled for label in review.labels}

    @property
    def open_labels(self) -> set[str]:
        return {label for review in self.open for label in review.labels}


class QueueBuilder:
    def __init__(
        self,
        dataset: Dataset,
        assessment: Assessment,
        *,
        queue_date: date | None = None,
        include_backlog: bool = False,
    ):
        self.dataset = dataset
        self.assessment = assessment
        self.queue_date = queue_date or (assessment.as_of + timedelta(days=1))
        self.include_backlog = include_backlog
        self.scope = self._scope_messages()
        self.channel_counts: dict[str, int] = {}
        for message in self.scope:
            self.channel_counts[message.channel] = self.channel_counts.get(message.channel, 0) + 1
        self.slots = self._build_slots()
        self.notes: list[str] = []
        self.decisions: list[QueueDecision] = []
        self._contexts: dict[str, FamilyContext] = {}
        self._preconsume_open_reviews()

    # -- scope and capacity ------------------------------------------------
    @staticmethod
    def _message_date(message: Message, fallback: date) -> date:
        return message.submitted_at.date or fallback

    def _scope_messages(self) -> list[Message]:
        until = self.assessment.messages_until_as_of()
        if self.include_backlog:
            return until
        return [
            message
            for message in until
            if self._message_date(message, self.assessment.as_of) == self.assessment.as_of
        ]

    def _build_slots(self) -> list[ShiftSlot]:
        slots: list[ShiftSlot] = []
        for capacity in self.dataset.capacities:
            default_start, default_end = DEFAULT_SHIFT_WINDOWS.get(
                capacity.shift, (time(0, 0), time(23, 59))
            )
            slots.append(
                ShiftSlot(
                    team=capacity.team,
                    shift=capacity.shift,
                    capacity=capacity.capacity,
                    start=capacity.start_time or default_start,
                    end=capacity.end_time or default_end,
                )
            )
        slots.sort(key=lambda item: (item.start, item.end, item.team, item.shift))
        return slots

    def _preconsume_open_reviews(self) -> None:
        """Unsettled reviews on today's messages already occupy seats."""

        scope_ids = {message.message_id for message in self.scope}
        for review in self.dataset.reviews:
            if review.closed and review.closed_at.at_or_before(
                self.assessment.as_of_cutoff, self.assessment.as_of
            ):
                continue
            if review.message_id not in scope_ids:
                continue
            teams: set[str] = set()
            for label in review.labels:
                teams.update(self.assessment.teams_for(label))
            if not teams:
                self.notes.append(f"open review {review.review_id} has no owning team")
                continue
            for team in sorted(teams):
                slot = self._next_slot(team)
                if slot is None:
                    self.notes.append(f"open review {review.review_id} exceeds {team} capacity")
                    continue
                slot.used += 1
                slot.preexisting += 1

    def _next_slot(self, team: str) -> ShiftSlot | None:
        for slot in self.slots:
            if slot.team == team and slot.used < slot.capacity:
                return slot
        return None

    # -- family context ----------------------------------------------------
    def _context(self, family: Family) -> FamilyContext:
        cached = self._contexts.get(family.family_id)
        if cached is not None:
            return cached
        settled: list[Review] = []
        open_reviews: list[Review] = []
        mandatory = False
        for member in family.members:
            if member.known_lead:
                mandatory = True
            for review in self.dataset.reviews_for(member.message_id):
                if MANDATORY_LABELS & set(review.labels):
                    mandatory = True
                if review.closed and review.closed_at.at_or_before(
                    self.assessment.as_of_cutoff, self.assessment.as_of
                ):
                    settled.append(review)
                else:
                    open_reviews.append(review)
        context = FamilyContext(family, settled, open_reviews, mandatory)
        self._contexts[family.family_id] = context
        return context

    def _is_novel(self, message: Message) -> tuple[bool, str]:
        novelty = self.assessment.novelty.get(message.message_id)
        if novelty and novelty.novel:
            return True, novelty.reason
        return False, ""

    def _high_impact(self, family: Family, channel: str) -> tuple[bool, bool]:
        top = max((member.impact for member in family.members), key=impact_rank)
        if impact_rank(top) < 3:
            return False, False
        total = max(len(self.scope), 1)
        share = self.channel_counts.get(channel, 0) / total
        return True, share <= 0.25

    # -- candidate labels --------------------------------------------------
    def _candidates(self, message: Message, context: FamilyContext) -> list[Candidate]:
        found: dict[str, Candidate] = {}

        def add(label: str, source: str, score: float) -> None:
            if label not in self.assessment.assessments:
                return
            existing = found.get(label)
            if existing is None or _source_rank(source, score) > _source_rank(
                existing.source, existing.score
            ):
                found[label] = Candidate(label, source, score)

        for review in context.settled:
            for label in review.labels:
                add(label, "review_closed", 1.0)
        for review in context.open:
            for label in review.labels:
                add(label, "review_open", 1.0)

        for label, score in self._model_scores(message):
            add(label, "model", score)

        if not found:
            for candidate in self._similar_labels(message):
                add(candidate.label, candidate.source, candidate.score)
        if not found:
            for candidate in self._triage_fallback(message):
                add(candidate.label, candidate.source, candidate.score)

        return sorted(found.values(), key=lambda item: (-item.score, item.label))

    def _model_scores(self, message: Message) -> list[tuple[str, float]]:
        result = []
        for label in self.assessment.label_catalog:
            item = self.assessment.assessments[label]
            if item.threshold_evidence == "insufficient":
                continue
            score = self.assessment.score(label, message.text)
            if score >= item.threshold:
                result.append((label, score))
        return result

    def _similar_labels(self, message: Message) -> list[Candidate]:
        """Nearest neighbour over reviewed messages; settled outranks open."""

        body = canonical_text(message.text)
        if not body:
            return []
        best: dict[str, Candidate] = {}
        for review in self.dataset.reviews:
            source_message = self.dataset.message(review.message_id)
            if source_message is None or source_message.message_id == message.message_id:
                continue
            other = canonical_text(source_message.text)
            if not other:
                continue
            score = containment(body, other)
            if score < SIMILARITY_MIN_SCORE:
                continue
            settled = review.closed and review.closed_at.at_or_before(
                self.assessment.as_of_cutoff, self.assessment.as_of
            )
            source = "similar_closed" if settled else "similar_open_provisional"
            for label in review.labels:
                current = best.get(label)
                if current is None or _source_rank(source, score) > _source_rank(
                    current.source, current.score
                ):
                    best[label] = Candidate(label, source, round(score, 6))
        return sorted(best.values(), key=lambda item: (-item.score, item.label))

    def _triage_fallback(self, message: Message) -> list[Candidate]:
        """Last resort for a message that matches nothing.

        A hand-off must never silently drop a message.  Pick the highest-scoring
        label and mark it manual-only.  If that label has no owning team the
        decision is recorded as ``no_owner``; it is never pushed onto an
        unrelated team just because that team happens to have capacity.
        """

        scores = [
            (label, self.assessment.score(label, message.text))
            for label in self.assessment.label_catalog
        ]
        if not scores:
            return []
        label, score = sorted(scores, key=lambda item: (-item[1], item[0]))[0]
        return [Candidate(label, "manual_triage", round(score, 6))]

    # -- construction ------------------------------------------------------
    def build(self) -> list[QueueDecision]:
        # A novel member gets its own group: it must not inherit the family's
        # closed conclusion, and its identifiers must not be merged into the
        # family's shared seat.
        groups: dict[tuple[str, str, str], dict] = {}
        for message in self.scope:
            family = self.assessment.membership[message.message_id]
            context = self._context(family)
            novel, novelty_reason = self._is_novel(message)
            candidates = self._candidates(message, context)
            if not candidates:
                self._emit_unassigned(message, family, context)
                continue
            high, small_channel = self._high_impact(family, message.channel)
            novel_key = message.message_id if novel else ""

            for candidate in candidates:
                label = candidate.label
                teams = self.assessment.teams_for(label)
                if not teams:
                    self._emit_unassigned(message, family, context, labels=[label])
                    continue
                item = self.assessment.assessments[label]
                routable = (
                    item.may_route(message.channel)
                    and candidate.source not in MANUAL_ONLY_SOURCES
                    and candidate.source not in REVIEW_SOURCES
                )
                for team in teams:
                    key = (family.family_id, team, novel_key)
                    group = groups.setdefault(
                        key, self._new_group(family, team, context, message)
                    )
                    group["messages"].add(message.message_id)
                    group["sources"][label] = candidate.source
                    group["scores"][label] = max(group["scores"].get(label, 0.0), candidate.score)
                    if candidate.source in PROVISIONAL_SOURCES:
                        group["provisional"] = True
                    if novel:
                        group["novel"] = True
                        group["novelty_reasons"].add(novelty_reason or "novel_substantive_appeal")

                    if label in context.closed_labels and not novel and not context.mandatory:
                        group["covered"].add(label)
                        continue
                    if label in context.open_labels and not novel and not context.mandatory:
                        group["open"].add(label)
                        continue
                    if (
                        routable
                        and not context.mandatory
                        and not high
                        and not novel
                    ):
                        group["auto"].add(label)
                        continue
                    group["seat"].add(label)

        for key in sorted(groups):
            self._emit(groups[key])

        self._emit_audit()
        self.decisions.sort(key=lambda item: item.priority_key())
        self._allocate()
        return self.decisions

    def _new_group(
        self, family: Family, team: str, context: FamilyContext, message: Message
    ) -> dict:
        return {
            "family": family,
            "team": team,
            "context": context,
            # The channel recorded on the decision must be the channel of a
            # message that actually produced this request; a shared family seat
            # otherwise borrows whichever member happened to be first.
            "channel": message.channel,
            "messages": set(),
            "sources": {},
            "scores": {},
            "seat": set(),
            "auto": set(),
            "covered": set(),
            "open": set(),
            "novel": False,
            "novelty_reasons": set(),
            "provisional": False,
        }

    def _emit(self, group: dict) -> None:
        family: Family = group["family"]
        team: str = group["team"]
        context: FamilyContext = group["context"]
        messages = self._unique_messages(group["messages"])
        if not messages:
            return
        representative = min(
            messages,
            key=lambda item: (
                0 if item.channel == group["channel"] else 1,
                item.submitted_at.sort_key(),
                item.message_id,
            ),
        )
        channel = group["channel"]
        high, small_channel = self._high_impact(family, channel)
        labels = sorted(group["seat"] | group["auto"] | group["covered"] | group["open"])
        seat_labels = sorted(group["seat"])
        reasons: set[str] = set()
        limitations: list[str] = []

        if context.mandatory:
            reasons.add("mandatory_known_lead")
            priority = PRIORITY_MANDATORY
        elif high:
            reasons.add("high_impact")
            if small_channel:
                reasons.add("small_channel_high_impact")
            priority = PRIORITY_HIGH_IMPACT
        elif group["novel"]:
            priority = PRIORITY_NOVEL
        else:
            priority = PRIORITY_NORMAL

        if group["novel"]:
            reasons.add("novel_substantive_appeal")
            reasons.update(group["novelty_reasons"])
        if group["covered"]:
            reasons.add("covered_by_closed_review")
        if group["open"]:
            reasons.add("already_open_review")
        if group["provisional"]:
            reasons.add("provisional_label_from_unsettled_review")
        if any(group["sources"].get(label) == "manual_triage" for label in group["seat"]):
            reasons.add("manual_triage_no_match")
        if len(messages) > 1:
            reasons.add("duplicate_family_shared_review")
        if len(seat_labels) > 1:
            reasons.add("multi_label")

        if group["seat"] and not context.mandatory and not high and not group["novel"]:
            if any(
                group["sources"].get(label) in PROVISIONAL_SOURCES for label in group["seat"]
            ):
                reasons.add("low_confidence_manual_triage")

        for label in labels:
            item = self.assessment.assessments.get(label)
            if item is None:
                limitations.append(f"unknown_label:{label}")
                continue
            limitations.extend(item.limitations)
            channel_item = item.channels.get(channel)
            if channel_item is not None:
                limitations.extend(channel_item.limitation)
                for reason in channel_item.reasons:
                    limitations.append(f"channel:{reason}")
            if not item.may_route(channel):
                limitations.append(f"label_not_approved_for_channel:{label}")

        if group["auto"] and group["seat"]:
            action = ACTION_QUEUED_MANUAL
        elif group["auto"]:
            action = ACTION_AUTO_ROUTED
            reasons.add("model_routed")
            priority = PRIORITY_AUDIT
        elif group["seat"]:
            action = ACTION_QUEUED_MANUAL
        elif group["open"]:
            action = ACTION_ALREADY_OPEN
        elif context.mandatory:
            # A mandatory lead is never merely covered or left open.
            action = ACTION_QUEUED_MANUAL
            reasons.add("mandatory_recheck")
        elif group["covered"] and len(messages) > 1:
            action = ACTION_COVERED
        elif context.settled and not group["novel"]:
            # Every member of this family was already reviewed and settled.
            action = ACTION_COVERED
        else:
            # A single unreviewed message must not be labelled as covered just
            # because some candidate label comes from elsewhere.
            action = ACTION_QUEUED_MANUAL
            reasons.add("manual_intake_required")

        if action == ACTION_QUEUED_MANUAL:
            for label in seat_labels:
                if not self.assessment.assessments[label].may_route(channel):
                    reasons.add("manual_intake_unverified")

        message_ids = sorted({message.message_id for message in messages})
        decision = QueueDecision(
            queue_date=self.queue_date.isoformat(),
            message_ids=message_ids,
            representative_id=representative.message_id,
            family_id=family.family_id,
            team=team,
            shift="",
            labels=labels,
            action=action,
            priority=priority,
            mandatory=context.mandatory,
            impact=self._family_impact(family),
            novel=bool(group["novel"]),
            reasons=sorted(reasons),
            limitations=sorted(set(limitations)),
            decision_version=self.assessment.decision_version,
            model_version=self._model_version(labels),
            policy_version=self.assessment.policy.version,
            channel=channel,
            submitted_at=representative.submitted_at.raw,
            label_sources={
                label: group["sources"].get(label, "review_closed") for label in labels
            },
            scores={
                label: round(group["scores"].get(label, 0.0), 6) for label in labels
            },
        )
        self.decisions.append(decision)

    def _emit_unassigned(
        self,
        message: Message,
        family: Family,
        context: FamilyContext,
        labels: Sequence[str] = (),
    ) -> None:
        self.decisions.append(
            QueueDecision(
                queue_date=self.queue_date.isoformat(),
                message_ids=[message.message_id],
                representative_id=message.message_id,
                family_id=family.family_id,
                team="",
                shift="",
                labels=list(labels),
                action=ACTION_NO_OWNER,
                priority=PRIORITY_NORMAL if not context.mandatory else PRIORITY_MANDATORY,
                mandatory=context.mandatory,
                impact=self._family_impact(family),
                novel=self._is_novel(message)[0],
                reasons=["no_team_assigns_label"] if labels else ["no_confident_label"],
                limitations=sorted(
                    {
                        f"label_not_approved_for_channel:{label}"
                        for label in labels
                        if not self.assessment.assessments[label].may_route(message.channel)
                    }
                ),
                decision_version=self.assessment.decision_version,
                model_version=self._model_version(labels),
                policy_version=self.assessment.policy.version,
                channel=message.channel,
                submitted_at=message.submitted_at.raw,
                label_sources={},
                scores={},
            )
        )

    def _emit_audit(self) -> None:
        seen: set[str] = set()
        for decision in self.decisions:
            seen.update(decision.message_ids)
        for message in self.scope:
            if message.message_id in seen:
                continue
            family = self.assessment.membership[message.message_id]
            context = self._context(family)
            self._emit_unassigned(message, family, context)

    def _allocate(self) -> None:
        for decision in self.decisions:
            if decision.action != ACTION_QUEUED_MANUAL:
                if decision.action == ACTION_AUTO_ROUTED:
                    decision.remaining_before = None
                continue
            slot = self._next_slot(decision.team)
            if slot is None:
                decision.action = (
                    ACTION_OVERFLOW if decision.mandatory else ACTION_DEFERRED
                )
                decision.reasons = sorted(
                    set(decision.reasons)
                    | ({"mandatory_over_capacity"} if decision.mandatory else {"capacity_exhausted"})
                )
                decision.remaining_before = 0
                decision.remaining_after = 0
                continue
            decision.remaining_before = slot.remaining
            slot.used += 1
            decision.shift = slot.shift
            decision.remaining_after = slot.remaining
            decision.reasons = sorted(set(decision.reasons) | {"shift_window"})

    # -- helpers -----------------------------------------------------------
    def _unique_messages(self, message_ids: Sequence[str]) -> list[Message]:
        messages = [self.dataset.message(message_id) for message_id in message_ids]
        ordered = sorted(
            (message for message in messages if message is not None),
            key=lambda item: (item.submitted_at.sort_key(), item.message_id),
        )
        return ordered

    def _family_impact(self, family: Family) -> str:
        best = max(family.members, key=lambda item: (impact_rank(item.impact), item.message_id))
        return best.impact

    def _model_version(self, labels: Sequence[str]) -> str:
        if labels and labels[0] in self.assessment.assessments:
            return self.assessment.assessments[labels[0]].model_version
        return ""

    # -- reporting ---------------------------------------------------------
    def shift_report(self) -> list[dict]:
        return [slot.to_dict() for slot in self.slots]

    def manual_intake_summary(self) -> list[dict]:
        result = []
        for label in self.assessment.label_catalog:
            item = self.assessment.assessments[label]
            if item.status == "approved" and all(
                channel.status == "approved" for channel in item.channels.values()
            ):
                continue
            channel_reasons = sorted(
                {
                    f"{channel}:{reason}"
                    for channel, value in item.channels.items()
                    for reason in value.reasons
                }
            )
            result.append(
                {
                    "label": label,
                    "status": item.status,
                    "reasons": sorted(set(item.reasons) | set(channel_reasons)),
                    "limitations": item.limitations,
                    "channels": {
                        channel: {
                            "status": value.status,
                            "reasons": value.reasons,
                            "limitations": value.limitation,
                        }
                        for channel, value in sorted(item.channels.items())
                    },
                    "teams": item.teams,
                    "note": (
                        "no_recent_batch"
                        if "no_independent_recent_event_batch" in item.limitations
                        else "channels_not_yet_validated"
                        if item.status == "partial"
                        else ""
                    ),
                }
            )
        return result

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for decision in self.decisions:
            counts[decision.action] = counts.get(decision.action, 0) + 1
        return {
            "queue_date": self.queue_date.isoformat(),
            "decision_version": self.assessment.decision_version,
            "policy_version": self.assessment.policy.version,
            "scope_messages": len(self.scope),
            "actions": counts,
            "shifts": self.shift_report(),
            "manual_intake": self.manual_intake_summary(),
            "notes": list(self.notes),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }

    def csv_rows(self) -> list[dict]:
        return [decision.csv_row() for decision in self.decisions]


def _source_rank(source: str, score: float) -> tuple:
    order = {
        "review_closed": 5,
        "review_open": 4,
        "model": 3,
        "similar_closed": 2,
        "similar_open_provisional": 1,
    }
    return (order.get(source, 0), score)
