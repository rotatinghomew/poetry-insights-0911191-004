"""Per-label triage model, evidence splits and go-live gates.

Design rules enforced here:

* Every label gets its own model and its own evidence.  A global accuracy
  figure is never computed, let alone used to release a category.
* Training, threshold selection and the most recent event-batch validation
  are separated in time and by duplicate/forward family.
* Only settled reviews may be used.  An unclosed review is neither a positive
  nor a negative sample.
* Reviews chosen by a previous version of the tool are kept out of the
  approval metrics; they are reported separately so selection bias can be
  told apart from real drift between periods.
* A category/channel with too little independent recent evidence stays on a
  manual intake path.
* Channel migration is measured and surfaced as a limitation; it does not
  silently inherit approval from another channel.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from datetime import date, time
from typing import Iterable, Sequence

from .families import Family, annotate_families
from .schema import Dataset, Message, Review, Stamp, canonical_text, infer_as_of, snapshot_digest

TOOL_VERSION = "handoff-tool/1.0.0"
MODEL_VERSION = "nb-char-bigram/1"
POLICY_VERSION = "policy-v1"

IMPACT_RANK = {"高": 3, "high": 3, "紧急": 3, "中": 2, "medium": 2, "低": 1, "low": 1}


def impact_rank(value: str) -> int:
    return IMPACT_RANK.get((value or "").strip().lower(), 0)


@dataclass(frozen=True)
class Policy:
    """Go-live thresholds.  Deliberately explicit and per label."""

    min_validation_pos: int = 3
    min_validation_neg: int = 3
    min_channel_pos: int = 2
    min_channel_neg: int = 2
    min_precision: float = 0.8
    min_recall: float = 0.5
    max_tool_share: float = 0.5
    temporal_drift_alert: float = 0.25
    min_prevalence_samples: int = 5
    alpha: float = 1.0

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    @property
    def version(self) -> str:
        return f"{POLICY_VERSION}-{self.fingerprint()}"


@dataclass
class Example:
    family_id: str
    message_id: str
    text: str
    channel: str
    event_id: str
    labels: frozenset[str]
    provenance: str  # tool | organic | unknown
    submitted: Stamp

    def has(self, label: str) -> bool:
        return label in self.labels


@dataclass
class Split:
    name: str
    family_ids: list[str] = field(default_factory=list)
    examples: list[Example] = field(default_factory=list)
    cutoff: Stamp = field(default_factory=Stamp)
    period_start: Stamp = field(default_factory=Stamp)
    excluded_carryover: list[str] = field(default_factory=list)


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def support_pos(self) -> int:
        return self.tp + self.fn

    @property
    def support_neg(self) -> int:
        return self.tn + self.fp

    @property
    def support(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None:
            return None
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "support": self.support,
            "support_pos": self.support_pos,
            "support_neg": self.support_neg,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def tokenize(text: str) -> list[str]:
    value = canonical_text(text)
    if not value:
        return []
    if len(value) == 1:
        return [value]
    tokens = [value[index : index + 2] for index in range(len(value) - 1)]
    if len(value) <= 6:
        tokens.append(value)
    return tokens


class NaiveBayesModel:
    """Small, deterministic multinomial NB over character bigrams."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.pos: dict[str, int] = {}
        self.neg: dict[str, int] = {}
        self.pos_total = 0
        self.neg_total = 0
        self.pos_docs = 0
        self.neg_docs = 0
        self.vocab: set[str] = set()

    def fit(self, examples: Sequence[Example], label: str) -> "NaiveBayesModel":
        for example in examples:
            target = self.pos if example.has(label) else self.neg
            if example.has(label):
                self.pos_docs += 1
            else:
                self.neg_docs += 1
            for token in tokenize(example.text):
                target[token] = target.get(token, 0) + 1
                self.vocab.add(token)
        self.pos_total = sum(self.pos.values())
        self.neg_total = sum(self.neg.values())
        return self

    def log_odds(self, text: str) -> float:
        total_docs = self.pos_docs + self.neg_docs
        if total_docs == 0:
            return 0.0
        prior = math.log((self.pos_docs + self.alpha) / (self.neg_docs + self.alpha))
        vocab_size = max(len(self.vocab), 1)
        pos_denominator = self.pos_total + self.alpha * vocab_size
        neg_denominator = self.neg_total + self.alpha * vocab_size
        score = prior
        for token in tokenize(text):
            if token not in self.vocab:
                continue
            pos_prob = (self.pos.get(token, 0) + self.alpha) / pos_denominator
            neg_prob = (self.neg.get(token, 0) + self.alpha) / neg_denominator
            score += math.log(pos_prob) - math.log(neg_prob)
        return score

    def score(self, text: str) -> float:
        value = self.log_odds(text)
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exponent = math.exp(value)
        return exponent / (1.0 + exponent)


@dataclass
class ChannelAssessment:
    channel: str
    status: str  # approved | manual
    reasons: list[str]
    metrics: Metrics
    training_examples: int
    limitation: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "status": self.status,
            "reasons": list(self.reasons),
            "metrics": self.metrics.to_dict(),
            "training_examples": self.training_examples,
            "limitations": list(self.limitation),
        }


@dataclass
class LabelAssessment:
    label: str
    status: str  # approved | partial | manual
    reasons: list[str]
    limitations: list[str]
    threshold: float
    threshold_evidence: str
    validation: Metrics
    validation_all: Metrics
    organic_support_pos: int
    organic_support_neg: int
    tool_share_recent: float | None
    channels: dict[str, ChannelAssessment]
    teams: list[str]
    selection_diagnostic: dict
    model_version: str
    policy_version: str

    @property
    def approved_channels(self) -> list[str]:
        return sorted(
            channel for channel, item in self.channels.items() if item.status == "approved"
        )

    def may_route(self, channel: str) -> bool:
        item = self.channels.get(channel)
        return item is not None and item.status == "approved"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "status": self.status,
            "reasons": list(self.reasons),
            "limitations": list(self.limitations),
            "threshold": self.threshold,
            "threshold_evidence": self.threshold_evidence,
            "validation": self.validation.to_dict(),
            "validation_all": self.validation_all.to_dict(),
            "organic_support_pos": self.organic_support_pos,
            "organic_support_neg": self.organic_support_neg,
            "tool_share_recent": self.tool_share_recent,
            "approved_channels": self.approved_channels,
            "channels": {name: item.to_dict() for name, item in sorted(self.channels.items())},
            "teams": list(self.teams),
            "selection_diagnostic": self.selection_diagnostic,
            "model_version": self.model_version,
            "policy_version": self.policy_version,
        }


class Assessment:
    """Everything needed to explain and reproduce a go-live decision."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        as_of: date | None = None,
        policy: Policy | None = None,
        labels: Iterable[str] | None = None,
    ):
        self.dataset = dataset
        self.policy = policy or Policy()
        self.as_of = as_of or infer_as_of(dataset)
        self.as_of_cutoff = Stamp(self.as_of, time(23, 59, 59), f"{self.as_of} end-of-day")
        self.families, self.membership, self.novelty = annotate_families(dataset)
        self.label_catalog = self._catalog(labels)
        self.splits = self._build_splits()
        self.models: dict[str, NaiveBayesModel] = {}
        self.assessments: dict[str, LabelAssessment] = {}
        self._fit()

    # -- catalogue ---------------------------------------------------------
    def _catalog(self, labels: Iterable[str] | None) -> list[str]:
        found: list[str] = []
        # Open reviews contribute label names (so a destination can be named)
        # but never evidence (they are not settled).
        for review in self.dataset.reviews:
            for label in review.labels:
                if label not in found:
                    found.append(label)
        for capacity in self.dataset.capacities:
            for label in capacity.required_labels:
                if label not in found:
                    found.append(label)
        if labels is not None:
            for label in labels:
                if label not in found:
                    found.append(label)
        return sorted(found)

    def teams_for(self, label: str) -> list[str]:
        return sorted(
            {capacity.team for capacity in self.dataset.capacities if capacity.can_handle(label)}
        )

    def messages_until_as_of(self) -> list[Message]:
        return [
            message
            for message in self.dataset.messages
            if message.submitted_at.at_or_before(self.as_of_cutoff, self.as_of)
        ]

    # -- splits ------------------------------------------------------------
    def _batches(self) -> list[tuple[str, list[Message]]]:
        groups: dict[str, list[Message]] = {}
        for message in self.messages_until_as_of():
            key = message.event_id or f"date:{message.submitted_at.date or self.as_of}"
            groups.setdefault(key, []).append(message)
        result = []
        for key, messages in groups.items():
            messages.sort(key=lambda item: (item.submitted_at.sort_key(), item.message_id))
            result.append((key, messages))
        result.sort(key=lambda item: (item[1][0].submitted_at.sort_key(), item[0]))
        return result

    def _build_splits(self) -> dict[str, Split]:
        batches = self._batches()
        splits: dict[str, Split] = {
            "train": Split("train"),
            "threshold": Split("threshold"),
            "validation": Split("validation"),
        }
        if not batches:
            return {"train": splits["train"]}

        if len(batches) >= 3:
            phases = ["train"] * (len(batches) - 2) + ["threshold", "validation"]
        elif len(batches) == 2:
            phases = ["train", "validation"]
        else:
            phases = ["train"]

        # A family belongs to the phase of its first appearance.  Later
        # appearances are excluded there, never promoted into a newer phase.
        family_phase: dict[str, str] = {}
        for (key, messages), phase in zip(batches, phases):
            for message in messages:
                family_id = self.membership[message.message_id].family_id
                first = family_phase.get(family_id)
                if first is None:
                    family_phase[family_id] = phase
                    splits[phase].family_ids.append(family_id)
                elif first != phase:
                    splits[phase].excluded_carryover.append(message.message_id)

        if len(batches) >= 3:
            validation_batch = batches[-1]
            threshold_batch = batches[-2]
            splits["validation"].period_start = validation_batch[1][0].submitted_at
            splits["validation"].cutoff = self.as_of_cutoff
            splits["threshold"].period_start = threshold_batch[1][0].submitted_at
            splits["threshold"].cutoff = validation_batch[1][0].submitted_at
            splits["train"].cutoff = threshold_batch[1][0].submitted_at
            splits["train"].period_start = batches[0][1][0].submitted_at
        elif len(batches) == 2:
            validation_batch = batches[-1]
            splits["validation"].period_start = validation_batch[1][0].submitted_at
            splits["validation"].cutoff = self.as_of_cutoff
            splits["train"].cutoff = validation_batch[1][0].submitted_at
            splits["train"].period_start = batches[0][1][0].submitted_at
            splits.pop("threshold")
        else:
            splits["train"].cutoff = self.as_of_cutoff
            splits["train"].period_start = batches[0][1][0].submitted_at
            splits.pop("threshold")
            splits.pop("validation")

        for split in splits.values():
            self._attach_examples(split)
        return splits

    def _attach_examples(self, split: Split) -> None:
        examples: list[Example] = []
        for family_id in sorted(split.family_ids):
            family = self.families[family_id]
            reviews = self._settled_reviews_for_family(family, split.cutoff)
            if not reviews:
                continue
            labels: set[str] = set()
            provenance = "organic"
            for review in reviews:
                labels.update(review.labels)
                if review.selected_by_tool:
                    provenance = "tool"
                elif review.selection.strip().lower() in ("", "unknown") and provenance != "tool":
                    provenance = "unknown"
            representative = self._representative(family, split.cutoff)
            examples.append(
                Example(
                    family_id=family_id,
                    message_id=representative.message_id,
                    text=representative.text,
                    channel=representative.channel,
                    event_id=representative.event_id,
                    labels=frozenset(labels),
                    provenance=provenance,
                    submitted=representative.submitted_at,
                )
            )
        split.examples = examples

    def _settled_reviews_for_family(self, family: Family, cutoff: Stamp) -> list[Review]:
        reviews: list[Review] = []
        for member in family.members:
            for review in self.dataset.reviews_for(member.message_id):
                if not review.closed:
                    continue
                if not review.closed_at.at_or_before(cutoff, self.as_of):
                    continue
                reviews.append(review)
        return reviews

    def _representative(self, family: Family, cutoff: Stamp | None = None) -> Message:
        """Pick the message whose settled review best supports the labels.

        Any member of a forward family carries the same evidence, but choosing
        the one an operative actually settled keeps the audit trail honest.
        The settled review must also fall within this split's cutoff, otherwise
        a later settlement would leak into an earlier training phase.
        """

        cutoff = cutoff or self.as_of_cutoff

        def has_settled(member: Message) -> int:
            return 0 if any(
                review.closed
                and review.closed_at.at_or_before(cutoff, self.as_of)
                for review in self.dataset.reviews_for(member.message_id)
            ) else 1

        return min(
            family.members,
            key=lambda item: (
                has_settled(item),
                item.submitted_at.sort_key(),
                item.message_id,
            ),
        )

    # -- fitting -----------------------------------------------------------
    def _fit(self) -> None:
        train = self.splits["train"].examples
        threshold_examples = self.splits.get("threshold", Split("threshold")).examples
        for label in self.label_catalog:
            base_model = NaiveBayesModel(self.policy.alpha).fit(train, label)
            threshold, evidence = self._select_threshold(base_model, label, train, threshold_examples)
            final_examples = list(train)
            if evidence == "held_out_period":
                final_examples.extend(threshold_examples)
            final_model = NaiveBayesModel(self.policy.alpha).fit(final_examples, label)
            self.models[label] = final_model
            self.assessments[label] = self._assess_label(
                label, final_model, threshold, evidence, train, threshold_examples
            )

    def score(self, label: str, text: str) -> float:
        model = self.models.get(label)
        return model.score(text) if model else 0.0

    def _select_threshold(
        self,
        model: NaiveBayesModel,
        label: str,
        train: Sequence[Example],
        threshold_examples: Sequence[Example],
    ) -> tuple[float, str]:
        if threshold_examples:
            targets = [example.has(label) for example in threshold_examples]
            if any(targets) and not all(targets):
                scores = [(model.score(example.text), target) for example, target in zip(threshold_examples, targets)]
                return self._best_threshold(scores), "held_out_period"

        cv_scores = self._cv_scores(label, train)
        targets = [target for _, target in cv_scores]
        if targets and any(targets) and not all(targets):
            return self._best_threshold(cv_scores), "family_grouped_cv"
        return 0.5, "insufficient"

    def _cv_scores(self, label: str, train: Sequence[Example]) -> list[tuple[float, bool]]:
        """Out-of-fold scores; every family stays wholly inside one fold."""

        if len(train) < 3:
            return []
        folds: dict[int, list[Example]] = {}
        for example in train:
            index = int(hashlib.sha256(example.family_id.encode("utf-8")).hexdigest(), 16) % 3
            folds.setdefault(index, []).append(example)
        if len(folds) < 2:
            return []
        scored: list[tuple[float, bool]] = []
        for fold_index, held_out in folds.items():
            others = [item for index, group in folds.items() if index != fold_index for item in group]
            if not others:
                continue
            fold_model = NaiveBayesModel(self.policy.alpha).fit(others, label)
            for example in held_out:
                scored.append((fold_model.score(example.text), example.has(label)))
        return scored

    @staticmethod
    def _best_threshold(scores: Sequence[tuple[float, bool]]) -> float:
        candidates = sorted({score for score, _ in scores})
        if not candidates:
            return 0.5
        best_key = None
        best_threshold = 0.5
        for threshold in candidates:
            metrics = Metrics()
            for score, target in scores:
                _add(metrics, score >= threshold, target)
            if metrics.tp + metrics.fp == 0:
                continue
            key = (metrics.f1 or 0.0, metrics.precision or 0.0, metrics.recall or 0.0, -threshold)
            if best_key is None or key > best_key:
                best_key = key
                best_threshold = threshold
        return best_threshold

    # -- gates -------------------------------------------------------------
    def _assess_label(
        self,
        label: str,
        model: NaiveBayesModel,
        threshold: float,
        threshold_evidence: str,
        train: Sequence[Example],
        threshold_examples: Sequence[Example],
    ) -> LabelAssessment:
        policy = self.policy
        validation = self.splits.get("validation")
        reasons: list[str] = []
        limitations: list[str] = []
        all_metrics = Metrics()
        organic_metrics = Metrics()
        organic_pos = organic_neg = 0
        tool_recent = 0
        unknown_recent = 0
        channels: dict[str, ChannelAssessment] = {}

        if validation is None:
            reasons.append("insufficient_recent_batch")
            limitations.append("no_independent_recent_event_batch")
            return self._manual_assessment(
                label, threshold, threshold_evidence, all_metrics, reasons, limitations
            )

        per_channel: dict[str, Metrics] = {}
        per_channel_organic: dict[str, Metrics] = {}
        per_channel_train: dict[str, int] = {}
        for example in validation.examples:
            target = example.has(label)
            predicted = model.score(example.text) >= threshold
            _add(all_metrics, predicted, target)
            _add(per_channel.setdefault(example.channel, Metrics()), predicted, target)
            if example.provenance == "tool":
                tool_recent += 1
            elif example.provenance == "unknown":
                # Without a recorded source the sample cannot be separated
                # from tool selection, so it is reported but never used as
                # approval evidence.
                unknown_recent += 1
            else:
                _add(organic_metrics, predicted, target)
                _add(per_channel_organic.setdefault(example.channel, Metrics()), predicted, target)
                if target:
                    organic_pos += 1
                else:
                    organic_neg += 1
        for example in train:
            per_channel_train[example.channel] = per_channel_train.get(example.channel, 0) + 1

        tool_share = tool_recent / len(validation.examples) if validation.examples else None

        if not validation.examples:
            reasons.append("no_settled_recent_reviews")
            limitations.append("open_or_missing_reviews_cannot_serve_as_negatives")
        if organic_pos < policy.min_validation_pos:
            reasons.append("insufficient_independent_positives")
        if organic_neg < policy.min_validation_neg:
            reasons.append("insufficient_independent_negatives")
        if threshold_evidence == "insufficient":
            reasons.append("threshold_not_validated")
        elif threshold_evidence == "family_grouped_cv":
            limitations.append("threshold_selected_by_family_grouped_cv")
        if tool_share is not None and tool_share > policy.max_tool_share:
            reasons.append("selection_bias_confounded")
            limitations.append("recent_reviews_mostly_tool_selected")
        elif tool_recent:
            limitations.append("tool_selected_reviews_excluded_from_approval")
        if unknown_recent:
            limitations.append("review_selection_provenance_unknown")

        precision = organic_metrics.precision
        recall = organic_metrics.recall
        if precision is None:
            reasons.append("precision_unmeasured")
        elif precision < policy.min_precision:
            reasons.append("precision_below_gate")
        if recall is None:
            reasons.append("recall_unmeasured")
        elif recall < policy.min_recall:
            reasons.append("recall_below_gate")

        observed_channels = {message.channel for message in self.messages_until_as_of()}
        channels_to_check = sorted(
            set(per_channel) | set(per_channel_train) | observed_channels
        )
        for channel in channels_to_check:
            metrics = per_channel_organic.get(channel, Metrics())
            channel_reasons: list[str] = []
            channel_limitations: list[str] = []
            train_count = per_channel_train.get(channel, 0)
            if metrics.support == 0:
                channel_reasons.append("channel_unvalidated")
            else:
                if metrics.support_pos < policy.min_channel_pos:
                    channel_reasons.append("channel_insufficient_positives")
                if metrics.support_neg < policy.min_channel_neg:
                    channel_reasons.append("channel_insufficient_negatives")
                if metrics.precision is None or metrics.precision < policy.min_precision:
                    channel_reasons.append("channel_precision_below_gate")
                if metrics.recall is None or metrics.recall < policy.min_recall:
                    channel_reasons.append("channel_recall_below_gate")
            if train_count == 0:
                channel_limitations.append("channel_absent_from_training")
            channels[channel] = ChannelAssessment(
                channel=channel,
                status="manual" if channel_reasons else "approved",
                reasons=channel_reasons,
                metrics=metrics,
                training_examples=train_count,
                limitation=channel_limitations,
            )

        teams = self.teams_for(label)
        if not teams:
            reasons.append("no_team_handles_label")

        diagnostic = self._selection_diagnostic(label, validation, train, threshold_examples)
        if diagnostic["verdict"] == "insufficient":
            limitations.append("insufficient_to_separate_selection_bias_from_drift")
        else:
            if (
                diagnostic.get("selection_gap") is not None
                and abs(diagnostic["selection_gap"]) >= policy.temporal_drift_alert
            ):
                limitations.append("selection_bias_gap_observed")
            if (
                diagnostic.get("temporal_gap") is not None
                and abs(diagnostic["temporal_gap"]) >= policy.temporal_drift_alert
            ):
                limitations.append("temporal_drift_observed")
            if (
                diagnostic.get("channel_shift") is not None
                and diagnostic["channel_shift"] >= policy.temporal_drift_alert
            ):
                limitations.append("channel_migration_observed")

        if reasons:
            status = "manual"
        elif not channels:
            reasons.append("no_channel_evidence")
            status = "manual"
        elif all(item.status != "approved" for item in channels.values()):
            reasons.append("no_channel_approved")
            status = "manual"
        elif any(item.status != "approved" for item in channels.values()):
            status = "partial"
        else:
            status = "approved"

        return LabelAssessment(
            label=label,
            status=status,
            reasons=reasons,
            limitations=limitations,
            threshold=threshold,
            threshold_evidence=threshold_evidence,
            validation=organic_metrics,
            validation_all=all_metrics,
            organic_support_pos=organic_pos,
            organic_support_neg=organic_neg,
            tool_share_recent=tool_share,
            channels=channels,
            teams=teams,
            selection_diagnostic=diagnostic,
            model_version=MODEL_VERSION,
            policy_version=policy.version,
        )

    def _manual_assessment(
        self, label, threshold, evidence, metrics, reasons, limitations
    ) -> LabelAssessment:
        return LabelAssessment(
            label=label,
            status="manual",
            reasons=reasons,
            limitations=limitations,
            threshold=threshold,
            threshold_evidence=evidence,
            validation=Metrics(),
            validation_all=metrics,
            organic_support_pos=0,
            organic_support_neg=0,
            tool_share_recent=None,
            channels={},
            teams=self.teams_for(label),
            selection_diagnostic={"verdict": "insufficient", "reason": "no_recent_batch"},
            model_version=MODEL_VERSION,
            policy_version=self.policy.version,
        )

    def _selection_diagnostic(
        self,
        label: str,
        validation: Split,
        train: Sequence[Example],
        threshold_examples: Sequence[Example],
    ) -> dict:
        """Compare organic vs tool-selected and earlier vs recent.

        Selection gap = tool vs organic prevalence in the same period.
        Temporal gap = organic earlier vs organic recent prevalence.
        Channel shift = total variation distance of channel mixes.
        Only reported when each group has enough independent families.
        """

        recent = validation.examples
        earlier = list(train) + list(threshold_examples)
        recent_organic = [item for item in recent if item.provenance != "tool"]
        recent_tool = [item for item in recent if item.provenance == "tool"]
        earlier_organic = [item for item in earlier if item.provenance != "tool"]
        earlier_tool = [item for item in earlier if item.provenance == "tool"]

        def prevalence(examples):
            if not examples:
                return None
            return sum(1 for item in examples if item.has(label)) / len(examples)

        def channel_mix(examples):
            counts: dict[str, int] = {}
            for item in examples:
                counts[item.channel] = counts.get(item.channel, 0) + 1
            total = sum(counts.values())
            return {key: value / total for key, value in counts.items()} if total else {}

        minimum = self.policy.min_prevalence_samples
        result = {
            "recent_families": len(recent),
            "recent_tool_families": len(recent_tool),
            "recent_nontool_families": len(recent_organic),
            "earlier_nontool_families": len(earlier_organic),
            "earlier_tool_families": len(earlier_tool),
            "recent_tool_prevalence": prevalence(recent_tool),
            "recent_nontool_prevalence": prevalence(recent_organic),
            "earlier_nontool_prevalence": prevalence(earlier_organic),
            "selection_gap": None,
            "temporal_gap": None,
            "channel_shift": None,
            "verdict": "insufficient",
        }
        if (
            len(recent_tool) >= minimum
            and len(recent_organic) >= minimum
            and result["recent_tool_prevalence"] is not None
            and result["recent_nontool_prevalence"] is not None
        ):
            result["selection_gap"] = (
                result["recent_tool_prevalence"] - result["recent_nontool_prevalence"]
            )
        if (
            len(recent_organic) >= minimum
            and len(earlier_organic) >= minimum
            and result["recent_nontool_prevalence"] is not None
            and result["earlier_nontool_prevalence"] is not None
        ):
            result["temporal_gap"] = (
                result["recent_nontool_prevalence"] - result["earlier_nontool_prevalence"]
            )
        if len(recent_organic) >= minimum and len(earlier_organic) >= minimum:
            recent_mix = channel_mix(recent_organic)
            earlier_mix = channel_mix(earlier_organic)
            keys = set(recent_mix) | set(earlier_mix)
            result["channel_shift"] = 0.5 * sum(
                abs(recent_mix.get(key, 0.0) - earlier_mix.get(key, 0.0)) for key in keys
            )
        if (
            result["selection_gap"] is not None
            or result["temporal_gap"] is not None
            or result["channel_shift"] is not None
        ):
            result["verdict"] = "estimated"
        return result

    # -- public ------------------------------------------------------------
    @property
    def decision_version(self) -> str:
        snapshot = snapshot_digest(self.dataset, extra=self.as_of.isoformat())
        return f"{MODEL_VERSION}+{self.policy.version}+{snapshot[:10]}"

    def summary(self) -> dict:
        return {
            "tool_version": TOOL_VERSION,
            "model_version": MODEL_VERSION,
            "policy_version": self.policy.version,
            "decision_version": self.decision_version,
            "as_of": self.as_of.isoformat(),
            "labels": [self.assessments[label].to_dict() for label in self.label_catalog],
            "splits": {
                name: {
                    "families": len(split.family_ids),
                    "examples": len(split.examples),
                    "period_start": split.period_start.raw,
                    "cutoff": split.cutoff.raw,
                    "carryover_excluded": len(split.excluded_carryover),
                }
                for name, split in sorted(self.splits.items())
            },
            "warnings": list(self.dataset.warnings),
            "global_accuracy_note": (
                "No global accuracy is computed or used. Each category and channel "
                "is gated on its own independent evidence."
            ),
        }

    def label_report(self, label: str) -> dict:
        if label not in self.assessments:
            raise KeyError(f"unknown label: {label}")
        return self.assessments[label].to_dict()


def _add(metrics: Metrics, predicted: bool, target: bool) -> None:
    if predicted and target:
        metrics.tp += 1
    elif predicted and not target:
        metrics.fp += 1
    elif not predicted and target:
        metrics.fn += 1
    else:
        metrics.tn += 1
