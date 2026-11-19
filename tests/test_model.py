import tempfile
import unittest
from datetime import date
from pathlib import Path

from handoff.model import Assessment
from handoff.schema import load_dataset

from tests.factories import capacity, dense_dataset, message, review, write_dataset


def _dense_dataset(directory: Path) -> Path:
    return dense_dataset(directory)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data = _dense_dataset(Path(self.temporary.name))
        self.dataset = load_dataset(self.data)

    def test_validation_is_the_most_recent_event_only(self):
        assessment = Assessment(self.dataset, as_of=date(2026, 11, 13))
        self.assertIn("validation", assessment.splits)
        validation_events = {
            assessment.dataset.message(example.message_id).event_id
            for example in assessment.splits["validation"].examples
        }
        self.assertEqual(validation_events, {"E3"})

    def test_open_review_is_not_a_negative_sample(self):
        with (self.data / "reviews.csv").open(encoding="utf-8") as source:
            rows = source.read()
        rows = rows.replace("2026-11-13 12:02", "")
        (self.data / "reviews.csv").write_text(rows, encoding="utf-8")
        dataset = load_dataset(self.data)
        assessment = Assessment(dataset, as_of=date(2026, 11, 13))
        family_ids = set(assessment.splits["validation"].family_ids)
        example_families = {example.family_id for example in assessment.splits["validation"].examples}
        self.assertTrue(family_ids - example_families, "the open family must be excluded")

    def test_families_never_cross_splits(self):
        assessment = Assessment(self.dataset, as_of=date(2026, 11, 13))
        seen: dict[str, str] = {}
        for name, split in assessment.splits.items():
            for family_id in split.family_ids:
                self.assertNotIn(family_id, seen, f"{family_id} appears in {seen.get(family_id)} and {name}")
                seen[family_id] = name

    def test_later_family_member_does_not_enter_newer_split(self):
        messages = [
            message("A", "E1", "2026-11-11 10:00", "建议增加字幕"),
            message("B", "E1", "2026-11-11 10:05", "建议增加字幕（转发）"),
            message("C", "E2", "2026-11-12 10:00", "场地坡道太窄"),
            message("D", "E3", "2026-11-13 10:00", "轮椅区域被围挡"),
        ]
        reviews = [
            review("V1", "A", "内容无障碍", "2026-11-11 12:00"),
            review("V2", "C", "场地指引", "2026-11-12 12:00"),
            review("V3", "D", "场地指引", "2026-11-13 12:00"),
        ]
        capacities = [capacity("内容组", "早班", 2, "内容无障碍"), capacity("现场组", "早班", 2, "场地指引")]
        with tempfile.TemporaryDirectory() as temporary:
            data = write_dataset(
                Path(temporary), messages=messages, reviews=reviews, capacities=capacities
            )
            assessment = Assessment(load_dataset(data), as_of=date(2026, 11, 13))
        train_families = {
            assessment.membership[example.message_id].family_id
            for example in assessment.splits["train"].examples
        }
        validation_families = {
            assessment.membership[example.message_id].family_id
            for example in assessment.splits["validation"].examples
        }
        self.assertTrue(train_families.isdisjoint(validation_families))


class GoLiveGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data = _dense_dataset(Path(self.temporary.name))

    def test_sparse_recent_category_stays_manual(self):
        dataset = load_dataset(self.data)
        assessment = Assessment(dataset, as_of=date(2026, 11, 13))
        sparse = assessment.assessments["教学建议"]
        self.assertNotEqual(sparse.status, "approved")
        self.assertIn("insufficient_independent_positives", sparse.reasons)

    def test_validated_channel_can_pass_while_another_channel_stays_manual(self):
        dataset = load_dataset(self.data)
        assessment = Assessment(dataset, as_of=date(2026, 11, 13))
        item = assessment.assessments["场地指引"]
        self.assertEqual(item.channels["现场"].status, "approved")
        self.assertEqual(item.channels["线上"].status, "manual")
        self.assertIn("channel_insufficient_positives", item.channels["线上"].reasons)
        self.assertEqual(item.status, "partial")
        self.assertFalse(item.may_route("线上"))
        self.assertTrue(item.may_route("现场"))

    def test_no_global_accuracy_is_reported(self):
        dataset = load_dataset(self.data)
        summary = Assessment(dataset, as_of=date(2026, 11, 13)).summary()
        self.assertNotIn("accuracy", summary)
        self.assertIn("global_accuracy_note", summary)

    def test_tool_selected_reviews_are_excluded_and_confound_is_flagged(self):
        with (self.data / "reviews.csv").open(encoding="utf-8") as source:
            rows = source.read()
        rows = rows.replace("organic", "tool-v0")
        (self.data / "reviews.csv").write_text(rows, encoding="utf-8")
        dataset = load_dataset(self.data)
        dataset.reviews  # reviews are immutable dataclasses; reload after edit
        assessment = Assessment(dataset, as_of=date(2026, 11, 13))
        for label in assessment.label_catalog:
            item = assessment.assessments[label]
            self.assertNotEqual(item.status, "approved", label)
            self.assertIn("selection_bias_confounded", item.reasons)

    def test_no_recent_batch_means_manual(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = write_dataset(
                Path(temporary),
                messages=[message("A", "E1", "2026-11-11 10:00", "建议增加字幕")],
                reviews=[review("V1", "A", "内容无障碍", "2026-11-11 12:00")],
                capacities=[capacity("内容组", "早班", 2, "内容无障碍")],
            )
            dataset = load_dataset(data)
            assessment = Assessment(dataset, as_of=date(2026, 11, 11))
        item = assessment.assessments["内容无障碍"]
        self.assertEqual(item.status, "manual")
        self.assertIn("insufficient_recent_batch", item.reasons)


if __name__ == "__main__":
    unittest.main()
