import unittest

from nlpaug.augmenter.word.context_word_embs import fix_punctuation_spacing


class TestPunctuationSpacing(unittest.TestCase):
    """Regression test for issue #325.

    ContextualWordEmbsAug decoded "(group)" as "( group )" with extra
    spaces around brackets/punctuation.
    """

    def test_brackets(self):
        self.assertEqual(
            'separate those (group) into Separate strings.',
            fix_punctuation_spacing('separate those ( group ) into Separate strings.'))
        self.assertEqual(
            'a [b] c {d} e',
            fix_punctuation_spacing('a [ b ] c { d } e'))

    def test_punctuation(self):
        self.assertEqual(
            'Hello, world!',
            fix_punctuation_spacing('Hello , world !'))
        self.assertEqual(
            'Wait: what? Yes; no... 100%',
            fix_punctuation_spacing('Wait : what ? Yes ; no . . . 100 %'))

    def test_issue_example_shape(self):
        # Full sentence from #325 (augmented words differ, spacing must not)
        decoded = 'Our goal is to separate those ( group ) into Separate strings .'
        fixed = fix_punctuation_spacing(decoded)
        self.assertNotIn('( ', fixed)
        self.assertNotIn(' )', fixed)
        self.assertNotIn(' .', fixed)
        self.assertEqual(
            'Our goal is to separate those (group) into Separate strings.',
            fixed)

    def test_conservative_no_quote_or_whitespace_changes(self):
        # Quotes and intentional double spaces are left alone (see #323)
        text = "don ' t  keep  double  spaces"
        self.assertEqual(text, fix_punctuation_spacing(text))


if __name__ == '__main__':
    unittest.main()
