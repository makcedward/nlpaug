import unittest

from nlpaug.model.word_stats.tfidf import TfIdf


class TestTfIdfTermFrequency(unittest.TestCase):
    """Regression test for issue #199.

    cal_tfidf must use TF=count(word)/len(tokens) like the UDA reference,
    not TF=1/len(tokens) for every occurrence.
    """

    def test_repeated_words_use_count_tf(self):
        m = TfIdf()
        m.w2idf = {'the': 1.0, 'cat': 2.0}
        tokens = ['the', 'cat', 'the']
        out = m.cal_tfidf([0, 1, 2], tokens, normalize=False)
        # the: 2/3*1.0, cat: 1/3*2.0, the: 2/3*1.0
        self.assertAlmostEqual(2 / 3, out[0])
        self.assertAlmostEqual(2 / 3, out[1])
        self.assertAlmostEqual(2 / 3, out[2])

    def test_distinct_words_unchanged(self):
        m = TfIdf()
        m.w2idf = {'a': 1.0, 'b': 2.0, 'c': 4.0}
        tokens = ['a', 'b', 'c']
        out = m.cal_tfidf([0, 1, 2], tokens, normalize=False)
        self.assertAlmostEqual(1 / 3, out[0])
        self.assertAlmostEqual(2 / 3, out[1])
        self.assertAlmostEqual(4 / 3, out[2])

    def test_subset_idxes_still_use_full_doc_counts(self):
        m = TfIdf()
        m.w2idf = {'the': 1.0, 'cat': 1.0}
        tokens = ['the', 'the', 'cat']
        out = m.cal_tfidf([0], tokens, normalize=False)
        self.assertAlmostEqual(2 / 3, out[0])


if __name__ == '__main__':
    unittest.main()
