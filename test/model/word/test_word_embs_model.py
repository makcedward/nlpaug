import unittest
import os
from dotenv import load_dotenv

import nlpaug.model.word_embs as nmw


class TestWordEmbsModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

    def test_bogus_fasttext_loading(self):
        test_file = os.path.join(os.environ.get("TEST_DIR"), 'res', 'text', 'bogus_fasttext.vec')
        expected_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        fasttext = nmw.Fasttext()
        fasttext.read(test_file)

        for word in fasttext.w2v:
            self.assertSequenceEqual(list(fasttext.w2v[word]), expected_vector)

        self.assertSequenceEqual(["test1", "test2", "test_3", "test 4", "test -> 5"], fasttext.get_vocab())

        self.assertEqual(len(fasttext.normalized_vectors), 5)
