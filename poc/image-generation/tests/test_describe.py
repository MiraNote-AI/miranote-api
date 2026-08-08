import unittest

import config


class DescribeQuestionTests(unittest.TestCase):
    """/describe grew a caller-supplied prompt so canvas mode can ask what a
    whole page looks like. The photo-import path must keep its old wording:
    it is what every existing ImageRef.summary was written with."""

    def test_default_is_the_photo_sentence(self):
        self.assertIn("one warm, concrete sentence", config.DESCRIBE_PROMPT)
        self.assertIn("Answer with the sentence only", config.DESCRIBE_PROMPT)

    def test_no_prompt_keeps_the_default(self):
        self.assertEqual(config.describe_question(None), config.DESCRIBE_PROMPT)

    def test_blank_prompt_keeps_the_default(self):
        self.assertEqual(config.describe_question("   "), config.DESCRIBE_PROMPT)

    def test_a_caller_question_replaces_the_default(self):
        self.assertEqual(
            config.describe_question("how does this page look?"),
            "how does this page look?",
        )

    def test_a_caller_question_is_trimmed(self):
        self.assertEqual(config.describe_question("  crowded?  "), "crowded?")


if __name__ == "__main__":
    unittest.main()
