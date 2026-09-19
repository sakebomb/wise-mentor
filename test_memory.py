import tempfile
import unittest
from pathlib import Path

from memory import MemoryStore, extract_facts, format_memory_block


class ExtractFactsTests(unittest.TestCase):
    def test_keeps_first_person_durable_claims(self):
        text = "Hey, my name is Sam and I like rockets. I play lacrosse."
        facts = extract_facts(text)
        joined = " ".join(facts).lower()
        self.assertTrue(any("sam" in f.lower() for f in facts))
        self.assertIn("rockets", joined)
        self.assertIn("lacrosse", joined)

    def test_drops_questions_and_chit_chat(self):
        self.assertEqual(extract_facts("what's 2+2?"), [])
        self.assertEqual(extract_facts("lol"), [])
        self.assertEqual(extract_facts(""), [])

    def test_strips_discord_mention_and_user_prefix(self):
        facts = extract_facts("<@1550704484030750741> I play chess")
        self.assertTrue(facts)
        self.assertTrue(all("<@" not in f for f in facts))
        facts2 = extract_facts("<@99>: I am a scout")
        self.assertTrue(any("scout" in f.lower() for f in facts2))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "memory.sqlite")

    def tearDown(self):
        self.tmp.cleanup()

    def test_remember_recall_round_trip(self):
        self.store.remember(1, "I like rockets")
        self.assertEqual(self.store.recall(1), ["I like rockets"])
        self.assertEqual(self.store.recall(2), [])

    def test_dedupes_casefold_and_skips_blank(self):
        self.store.remember(1, "I like Rockets")
        self.store.remember(1, "i like rockets")
        self.store.remember(1, "  ")
        self.assertEqual(self.store.recall(1), ["I like Rockets"])

    def test_forget_user_deletes_only_that_user(self):
        self.store.remember(1, "I like rockets")
        self.store.remember(2, "I play lacrosse")
        deleted = self.store.forget_user(1)
        self.assertEqual(deleted, 1)
        self.assertEqual(self.store.recall(1), [])
        self.assertEqual(self.store.recall(2), ["I play lacrosse"])

    def test_recall_caps_at_twenty_newest(self):
        for i in range(25):
            self.store.remember(1, f"I like item-{i}")
        facts = self.store.recall(1)
        self.assertEqual(len(facts), 20)
        self.assertEqual(facts[0], "I like item-24")
        self.assertEqual(facts[-1], "I like item-5")


class PromptBlockTests(unittest.TestCase):
    def test_empty_is_blank(self):
        self.assertEqual(format_memory_block([]), "")

    def test_lists_facts_for_the_model(self):
        block = format_memory_block(["I like rockets", "I play lacrosse"])
        self.assertIn("Known about this person", block)
        self.assertIn("- I like rockets", block)
        self.assertIn("- I play lacrosse", block)
        self.assertIn("do not list", block.lower())


if __name__ == "__main__":
    unittest.main()
