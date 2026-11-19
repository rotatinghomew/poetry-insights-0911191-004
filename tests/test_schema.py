import tempfile
import unittest
from pathlib import Path

from handoff.schema import DataError, load_dataset

from tests.factories import capacity, message, review, write_dataset


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _load(self, messages, reviews, capacities=None):
        data = write_dataset(
            self.root,
            messages=messages,
            reviews=reviews,
            capacities=capacities or [capacity("现场组", "早班", 1, "场地指引")],
        )
        return load_dataset(data)

    def test_duplicate_message_id_is_rejected(self):
        with self.assertRaises(DataError):
            self._load(
                [message("M1", "E1", "2026-11-13 10:00", "甲"), message("M1", "E1", "2026-11-13 10:01", "乙")],
                [],
            )

    def test_review_for_unknown_message_is_rejected(self):
        with self.assertRaises(DataError):
            self._load(
                [message("M1", "E1", "2026-11-13 10:00", "无障碍坡道入口")],
                [review("V1", "MISSING", "场地指引", "2026-11-13 11:00")],
            )

    def test_empty_cell_is_missing_not_negative(self):
        dataset = self._load(
            [message("M1", "E1", "2026-11-13 10:00", "无障碍坡道入口")],
            [review("V1", "M1", "场地指引", "")],
        )
        self.assertFalse(dataset.reviews[0].closed)
        self.assertFalse(dataset.messages[0].known_lead)

    def test_multi_label_cell_splits(self):
        dataset = self._load(
            [message("M1", "E1", "2026-11-13 10:00", "无障碍坡道入口并有字幕请求")],
            [review("V1", "M1", "场地指引;内容无障碍", "2026-11-13 11:00")],
        )
        self.assertEqual(dataset.reviews[0].labels, ("场地指引", "内容无障碍"))

    def test_review_closed_before_submission_is_flagged(self):
        dataset = self._load(
            [message("M1", "E1", "2026-11-13 10:00", "无障碍坡道入口")],
            [review("V1", "M1", "场地指引", "2026-11-12 09:00")],
        )
        self.assertTrue(
            any("closed before" in warning for warning in dataset.warnings), dataset.warnings
        )


if __name__ == "__main__":
    unittest.main()
