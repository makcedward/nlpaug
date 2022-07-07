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
        test_file = os.path.join(os.environ.get("PACKAGE_DIR"), 'res', 'text', 'bogus_fasttext.vec')

        # Change to not supporting incorrect format file after switching to use gensim package
        with self.assertRaises(Exception) as error:
            fasttext = nmw.Fasttext()
            fasttext.read(test_file)
        self.assertIn('cannot copy sequence with size 11 to array axis with dimension 10', str(error.exception))

        # for word in fasttext.get_vocab():
        #     self.assertSequenceEqual(list(fasttext.model[word]), expected_vector)

        # self.assertSequenceEqual(["test1", "test2", "test_3", "test 4", "test -> 5"], fasttext.get_vocab())

        # self.assertEqual(len(fasttext.get_vocab()), 5)
