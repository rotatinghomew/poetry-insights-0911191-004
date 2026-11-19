import unittest

from handoff.families import build_families, detect_novelty, family_novelty
from handoff.schema import Message, Stamp, canonical_text, load_dataset


def msg(message_id, text, *, event="E", channel="线上", submitted="10:00", **kwargs):
    return Message(
        message_id=message_id,
        channel=channel,
        event_id=event,
        submitted_at=Stamp(None, None, submitted),
        text=text,
        **kwargs,
    )


class CanonicalTextTests(unittest.TestCase):
    def test_politeness_and_punctuation_are_ignored(self):
        self.assertEqual(
            canonical_text("想请问一下：无障碍坡道入口在哪里？谢谢"),
            canonical_text("无障碍坡道入口在哪里"),
        )

    def test_forward_markers_are_ignored(self):
        for text in (
            "【转发】无障碍坡道入口在哪里",
            "转发：无障碍坡道入口在哪里",
            "FWD: 无障碍坡道入口在哪里",
            "无障碍坡道入口在哪里（转发）",
            "「无障碍坡道入口在哪里」",
        ):
            with self.subTest(text=text):
                self.assertEqual(canonical_text(text), canonical_text("无障碍坡道入口在哪里"))


class FamilyTests(unittest.TestCase):
    def test_fixture_forward_is_one_family(self):
        dataset = load_dataset("data")
        families = build_families(dataset)
        with_m01 = [family for family in families.values() if "M01" in family.message_ids]
        self.assertEqual(len(with_m01), 1)
        self.assertIn("M03", with_m01[0].message_ids)

    def test_reworded_intro_is_still_the_same_evidence(self):
        messages = [
            msg("A", "无障碍坡道入口在哪里"),
            msg("B", "想请问一下，无障碍坡道入口在哪里？"),
            msg("C", "转发：无障碍坡道入口在哪里"),
        ]
        dataset = _dataset(messages)
        families = build_families(dataset)
        self.assertEqual(len(families), 1)
        self.assertEqual(next(iter(families.values())).message_ids, ["A", "B", "C"])

    def test_verbatim_quote_is_not_novel(self):
        base = msg("A", "朗诵能否增加字幕")
        quote = msg("B", "引用：朗诵能否增加字幕")
        self.assertFalse(detect_novelty(quote, [base]).novel)

    def test_appended_request_is_novel(self):
        base = msg("A", "朗诵能否增加字幕")
        quote = msg("B", "朗诵能否增加字幕，另外希望增加手语翻译")
        novelty = detect_novelty(quote, [base])
        self.assertTrue(novelty.novel)
        self.assertIn("另外希望增加手语翻译", novelty.segments)

    def test_different_requests_sharing_words_do_not_merge(self):
        messages = [
            msg("A", "建议增加字幕", event="E1"),
            msg("B", "朗诵能否增加字幕", event="E2"),
        ]
        dataset = _dataset(messages)
        families = build_families(dataset)
        self.assertEqual(len(families), 2)

    def test_identical_template_across_events_is_one_family(self):
        messages = [
            msg("A", "引用范文：请增加无障碍字幕", event="E1"),
            msg("B", "请增加无障碍字幕", event="E2"),
        ]
        dataset = _dataset(messages)
        families = build_families(dataset)
        self.assertEqual(len(families), 1)

    def test_family_novelty_tracks_first_appearance(self):
        messages = [
            msg("A", "朗诵能否增加字幕", submitted="10:00"),
            msg("B", "朗诵能否增加字幕，另外希望增加手语翻译", submitted="10:05"),
        ]
        dataset = _dataset(messages)
        families = build_families(dataset)
        novelty = family_novelty(next(iter(families.values())))
        self.assertFalse(novelty["A"].novel)
        self.assertTrue(novelty["B"].novel)


def _dataset(messages):
    from handoff.schema import Dataset

    return Dataset(messages=messages, reviews=[], capacities=[])


if __name__ == "__main__":
    unittest.main()
