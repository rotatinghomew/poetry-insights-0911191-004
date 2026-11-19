import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from handoff.cli import main
from handoff.ledger import (
    LedgerError,
    append_decisions,
    decision_id,
    lookup_message,
    read_ledger,
)
from handoff.model import Assessment
from handoff.queue import QueueBuilder
from handoff.schema import load_dataset

from tests.factories import capacity, dense_dataset, message, review


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = dense_dataset(
            self.root / "data",
            extra_messages=[
                message("M1", "E3", "2026-11-13 15:00", "无障碍坡道东侧入口在哪里", channel="线上")
            ],
            capacities=[capacity("现场组", "早班", 2, "场地指引")],
        )
        self.dataset = load_dataset(self.data)
        self.assessment = Assessment(self.dataset, as_of=date(2026, 11, 13))
        self.builder = QueueBuilder(self.dataset, self.assessment)
        self.builder.build()
        self.ledger = self.root / "ledger.jsonl"

    def test_append_is_idempotent(self):
        added = append_decisions(self.ledger, self.builder.decisions, dataset=self.dataset)
        self.assertGreater(len(added), 0)
        again = append_decisions(self.ledger, self.builder.decisions, dataset=self.dataset)
        self.assertEqual(again, [])
        self.assertEqual(len(read_ledger(self.ledger)), len(added))

    def test_record_carries_version_reasons_and_limitations(self):
        append_decisions(self.ledger, self.builder.decisions, dataset=self.dataset)
        record = lookup_message(self.ledger, "M1")[0]
        self.assertEqual(record["decision_version"], self.assessment.decision_version)
        self.assertIn("model_version", record)
        self.assertIn("policy_version", record)
        self.assertTrue(record["reasons"])
        self.assertIn("inputs_digest", record)

    def test_handed_over_record_is_not_rewritten(self):
        append_decisions(self.ledger, self.builder.decisions, dataset=self.dataset)
        original = read_ledger(self.ledger)
        decision = self.builder.decisions[0]
        decision.action = "deferred_no_capacity"
        decision.reasons = ["capacity_exhausted"]
        with self.assertRaises(LedgerError):
            append_decisions(self.ledger, [decision], dataset=self.dataset)
        self.assertEqual(read_ledger(self.ledger), original)

    def test_later_review_does_not_change_old_record(self):
        append_decisions(self.ledger, self.builder.decisions, dataset=self.dataset)
        before = read_ledger(self.ledger)
        with (self.data / "reviews.csv").open("a", encoding="utf-8") as sink:
            sink.write("V-late,M1,U9,场地指引,2026-11-13 17:00,organic\n")
        dataset = load_dataset(self.data)
        assessment = Assessment(dataset, as_of=date(2026, 11, 13))
        builder = QueueBuilder(dataset, assessment)
        builder.build()
        append_decisions(self.ledger, builder.decisions, dataset=dataset)
        after = read_ledger(self.ledger)
        self.assertTrue(all(record in after for record in before))
        self.assertNotEqual(
            {record["decision_id"] for record in before},
            {record["decision_id"] for record in after},
        )
        self.assertNotEqual(
            {record["decision_version"] for record in before},
            {record["decision_version"] for record in after},
        )

    def test_decision_ids_are_stable(self):
        first = [decision_id(item) for item in self.builder.decisions]
        assessment = Assessment(load_dataset(self.data), as_of=date(2026, 11, 13))
        second_builder = QueueBuilder(assessment.dataset, assessment)
        second_builder.build()
        self.assertEqual(first, [decision_id(item) for item in second_builder.decisions])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = str(self.root / "data")
        dense_dataset(Path(self.data))
        self.out = self.root / "out"

    def _run(self, *argv):
        return main(list(argv))

    def test_assess_and_queue_and_explain(self):
        self.assertEqual(self._run("assess", "--data", self.data, "--out", str(self.out / "a.json")), 0)
        self.assertEqual(
            self._run(
                "queue",
                "--data",
                self.data,
                "--out",
                str(self.out / "q.json"),
                "--csv",
                str(self.out / "q.csv"),
                "--ledger",
                str(self.out / "l.jsonl"),
            ),
            0,
        )
        self.assertNotIn("overall", (self.out / "a.json").read_text(encoding="utf-8"))
        queue = json.loads((self.out / "q.json").read_text(encoding="utf-8"))
        self.assertIn("decisions", queue)
        self.assertTrue((self.out / "q.csv").exists())
        first_message = queue["decisions"][0]["message_ids"][0]
        self.assertEqual(
            self._run(
                "explain",
                first_message,
                "--ledger",
                str(self.out / "l.jsonl"),
                "--out",
                str(self.out / "e.json"),
            ),
            0,
        )
        explained = json.loads((self.out / "e.json").read_text(encoding="utf-8"))
        self.assertTrue(explained)

    def test_open_review_is_visible_in_assessment(self):
        self.assertEqual(self._run("assess", "--data", self.data, "--out", str(self.out / "a.json")), 0)
        payload = json.loads((self.out / "a.json").read_text(encoding="utf-8"))
        self.assertNotIn("accuracy", payload)


if __name__ == "__main__":
    unittest.main()
