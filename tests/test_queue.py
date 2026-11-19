import tempfile
import unittest
from datetime import date
from pathlib import Path

from handoff.model import Assessment
from handoff.queue import (
    ACTION_ALREADY_OPEN,
    ACTION_AUTO_ROUTED,
    ACTION_COVERED,
    ACTION_DEFERRED,
    ACTION_NO_OWNER,
    ACTION_OVERFLOW,
    ACTION_QUEUED_MANUAL,
    QueueBuilder,
)
from handoff.schema import load_dataset

from tests.factories import capacity, dense_dataset, message, review


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.as_of = date(2026, 11, 13)

    def _build(self, *, extra_messages=(), extra_reviews=(), capacities=None, **kwargs):
        data = dense_dataset(
            self.root / "data",
            extra_messages=list(extra_messages),
            extra_reviews=list(extra_reviews),
            capacities=capacities,
        )
        dataset = load_dataset(data)
        assessment = Assessment(dataset, as_of=kwargs.pop("as_of", self.as_of))
        builder = QueueBuilder(dataset, assessment, **kwargs)
        builder.build()
        return builder

    def _today(self, message_id, text, *, channel="现场", **extra):
        return message(message_id, "E3", "2026-11-13 15:00", text, channel=channel, **extra)

    def test_mandatory_lead_is_first_and_overflows_loudly(self):
        builder = self._build(
            extra_messages=[
                self._today("M1", "无障碍坡道入口在哪里", known_lead="yes"),
            ],
            capacities=[capacity("现场组", "早班", 0, "场地指引")],
        )
        mandatory = [item for item in builder.decisions if item.mandatory]
        self.assertTrue(mandatory)
        self.assertEqual(mandatory[0].priority, 0)
        self.assertEqual(mandatory[0].action, ACTION_OVERFLOW)
        self.assertIn("mandatory_over_capacity", mandatory[0].reasons)

    def test_one_full_team_does_not_block_another(self):
        builder = self._build(
            extra_messages=[
                self._today("M1", "无障碍坡道入口在哪里", channel="现场"),
                self._today("M2", "朗诵能否增加字幕", channel="线上"),
            ],
            capacities=[
                capacity("现场组", "早班", 1, "场地指引"),
                capacity("内容组", "早班", 1, "内容无障碍"),
            ],
        )
        seated = [item for item in builder.decisions if item.action == ACTION_QUEUED_MANUAL]
        self.assertEqual({item.team for item in seated}, {"现场组", "内容组"})
        self.assertFalse(any(item.action == ACTION_DEFERRED for item in seated))

    def test_duplicate_family_shares_one_seat(self):
        # 线上 场地指引 is not approved, so the family needs one manual seat.
        builder = self._build(
            extra_messages=[
                self._today("M1", "无障碍坡道东侧入口在哪里", channel="线上"),
                self._today("M2", "转发：无障碍坡道东侧入口在哪里", channel="线上"),
            ],
            capacities=[capacity("现场组", "早班", 1, "场地指引")],
        )
        seats = [
            item
            for item in builder.decisions
            if item.action == ACTION_QUEUED_MANUAL and "M1" in item.message_ids
        ]
        self.assertEqual(len(seats), 1)
        self.assertEqual(seats[0].message_ids, ["M1", "M2"])
        self.assertIn("duplicate_family_shared_review", seats[0].reasons)

    def test_novel_member_gets_its_own_seat(self):
        builder = self._build(
            extra_messages=[
                self._today("M1", "无障碍坡道东侧入口在哪里", channel="线上"),
                self._today("M2", "无障碍坡道东侧入口在哪里，另外轮渡时间能否提前", channel="线上"),
            ],
            capacities=[capacity("现场组", "早班", 2, "场地指引")],
        )
        novel = [item for item in builder.decisions if item.novel and "M2" in item.message_ids]
        seats = [item for item in novel if item.action == ACTION_QUEUED_MANUAL]
        self.assertEqual(len(seats), 1)
        self.assertEqual(seats[0].message_ids, ["M2"])
        self.assertIn("novel_substantive_appeal", seats[0].reasons)

    def test_novel_member_does_not_disappear_when_family_was_closed(self):
        builder = self._build(
            extra_messages=[
                self._today("M1", "无障碍坡道东侧入口在哪里", channel="线上"),
                self._today("M2", "无障碍坡道东侧入口在哪里，另外轮渡时间能否提前", channel="线上"),
            ],
            extra_reviews=[
                review("S1", "M1", "场地指引", "2026-11-13 16:00"),
            ],
            capacities=[capacity("现场组", "早班", 2, "场地指引")],
        )
        original = [item for item in builder.decisions if item.message_ids == ["M1"]]
        novel = [item for item in builder.decisions if item.novel and "M2" in item.message_ids]
        self.assertTrue(any(item.action == ACTION_COVERED for item in original))
        self.assertEqual(len(novel), 1)
        self.assertEqual(novel[0].message_ids, ["M2"])
        self.assertNotIn("M1", novel[0].message_ids)
        self.assertEqual(novel[0].action, ACTION_QUEUED_MANUAL)

    def test_closed_review_is_not_queued_again(self):
        builder = self._build(
            extra_messages=[self._today("M1", "无障碍坡道东侧入口在哪里", channel="线上")],
            extra_reviews=[review("S1", "M1", "场地指引", "2026-11-13 16:00")],
        )
        row = next(item for item in builder.decisions if "M1" in item.message_ids)
        self.assertEqual(row.action, ACTION_COVERED)

    def test_open_review_is_already_open(self):
        builder = self._build(
            extra_messages=[self._today("M1", "无障碍坡道东侧入口在哪里", channel="线上")],
            extra_reviews=[review("S1", "M1", "场地指引", "")],
        )
        row = next(item for item in builder.decisions if "M1" in item.message_ids)
        self.assertEqual(row.action, ACTION_ALREADY_OPEN)
        self.assertIn("provisional_label_from_unsettled_review", row.reasons)

    def test_high_impact_on_small_channel_is_promoted(self):
        builder = self._build(
            extra_messages=[
                self._today("C1", "轮椅被挡住无法进场", channel="校园", impact="高"),
                self._today("O1", "增加字幕", channel="线上", impact="低"),
                self._today("O2", "增加字幕第二场", channel="线上", impact="低"),
                self._today("O3", "增加字幕第三场", channel="线上", impact="低"),
                self._today("S1", "无障碍坡道东侧入口在哪里", channel="现场", impact="低"),
            ],
            capacities=[capacity("现场组", "早班", 5, "场地指引")],
        )
        campus = [item for item in builder.decisions if item.channel == "校园"]
        self.assertTrue(campus)
        self.assertIn("small_channel_high_impact", campus[0].reasons)
        self.assertEqual(campus[0].priority, 1)

    def test_shift_window_is_recorded(self):
        builder = self._build(
            extra_messages=[self._today("M1", "无障碍坡道入口在哪里")],
            capacities=[
                capacity("现场组", "早班", 1, "场地指引"),
                capacity("现场组", "晚班", 1, "场地指引"),
            ],
        )
        seat = next(item for item in builder.decisions if item.action == ACTION_QUEUED_MANUAL)
        self.assertEqual(seat.shift, "早班")
        self.assertIn("shift_window", seat.reasons)

    def test_unknown_message_is_still_handed_to_a_team(self):
        # A message matching no label must not vanish; it goes to manual
        # triage under an owning team instead of being silently dropped.
        builder = self._build(
            extra_messages=[self._today("X1", "想问问洗手间在哪个方向")],
        )
        row = next(item for item in builder.decisions if "X1" in item.message_ids)
        self.assertEqual(row.team, "现场组")
        self.assertEqual(row.action, ACTION_QUEUED_MANUAL)

    def test_label_without_team_is_no_owner(self):
        builder = self._build(
            extra_messages=[self._today("X1", "想问问洗手间在哪个方向")],
            capacities=[capacity("内容组", "早班", 2, "内容无障碍")],
        )
        self.assertTrue(
            any(
                item.action == ACTION_NO_OWNER or item.team == ""
                for item in builder.decisions
                if "X1" in item.message_ids
            )
        )

    def test_capacity_exhaustion_is_deferred_not_dropped(self):
        builder = self._build(
            extra_messages=[
                self._today("M1", "无障碍东侧坡道入口在哪里甲", channel="线上"),
                self._today("M2", "无障碍东侧坡道入口在哪里乙", channel="线上"),
                self._today("M3", "无障碍东侧坡道入口在哪里丙", channel="线上"),
                self._today("M4", "无障碍东侧坡道入口在哪里丁", channel="线上"),
            ],
            capacities=[capacity("现场组", "早班", 2, "场地指引")],
        )
        today = [item for item in builder.decisions if item.message_ids[0].startswith("M")]
        deferred = [item for item in today if item.action == ACTION_DEFERRED]
        self.assertEqual(len(deferred), 2)
        self.assertTrue(all("capacity_exhausted" in item.reasons for item in deferred))

    def test_every_scoped_message_has_a_destination(self):
        builder = self._build(
            extra_messages=[
                self._today("M1", "无障碍坡道东侧入口在哪里"),
                self._today("M2", "朗诵能否同步显示字幕", channel="线上"),
                self._today("M3", "这是什么意思", channel="校园"),
            ],
        )
        covered = set()
        for decision in builder.decisions:
            covered.update(decision.message_ids)
        for message_id in ("M1", "M2", "M3"):
            self.assertIn(message_id, covered)


if __name__ == "__main__":
    unittest.main()


class ExplainabilityTests(unittest.TestCase):
    """The operational questions must be answerable from the output alone."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _builder(self):
        data = dense_dataset(
            self.root / "data",
            extra_messages=[
                message("T1", "E3", "2026-11-13 15:00", "无障碍坡道东侧入口在哪里", channel="线上"),
                message("T2", "E3", "2026-11-13 15:00", "无障碍坡道东侧入口在哪里", channel="线上"),
            ],
        )
        dataset = load_dataset(data)
        assessment = Assessment(dataset, as_of=date(2026, 11, 13))
        builder = QueueBuilder(dataset, assessment)
        builder.build()
        return assessment, builder

    def test_each_destination_names_its_version_and_why(self):
        assessment, builder = self._builder()
        for decision in builder.decisions:
            self.assertEqual(decision.decision_version, assessment.decision_version)
            self.assertEqual(decision.policy_version, assessment.policy.version)
            self.assertTrue(decision.reasons, decision.message_ids)
            self.assertTrue(decision.label_sources, decision.message_ids)

    def test_manual_intake_lists_every_unreleased_category(self):
        assessment, builder = self._builder()
        manual = builder.manual_intake_summary()
        reported = {item["label"] for item in manual}
        unreleased = {
            label
            for label in assessment.label_catalog
            if assessment.assessments[label].status != "approved"
        }
        self.assertTrue(unreleased.issubset(reported))
        for item in manual:
            self.assertTrue(item["reasons"] or item["limitations"], item["label"])

    def test_reasons_explain_deferral_and_coverage(self):
        assessment, builder = self._builder()
        for decision in builder.decisions:
            if decision.action == ACTION_AUTO_ROUTED:
                self.assertIn("model_routed", decision.reasons)
            if decision.action == ACTION_COVERED:
                self.assertIn("covered_by_closed_review", decision.reasons)
            if decision.action == ACTION_DEFERRED:
                self.assertIn("capacity_exhausted", decision.reasons)
