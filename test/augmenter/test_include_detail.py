import inspect
import unittest

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw


class TestIncludeDetail(unittest.TestCase):
    def test_change_log_augmenters_expose_include_detail(self):
        augmenter_classes = [
            nac.KeyboardAug,
            nac.OcrAug,
            nac.RandomCharAug,
            naw.AntonymAug,
            naw.RandomWordAug,
            naw.ReservedAug,
            naw.SpellingAug,
            naw.SplitAug,
            naw.SynonymAug,
            naw.TfIdfAug,
            naw.WordEmbsAug,
        ]

        for augmenter_class in augmenter_classes:
            with self.subTest(augmenter=augmenter_class.__name__):
                parameter = inspect.signature(augmenter_class).parameters['include_detail']
                self.assertFalse(parameter.default)

    def test_include_detail_returns_text_and_change_logs(self):
        aug = naw.RandomWordAug(
            action='delete', target_words=['quick'], aug_p=1.0, include_detail=True
        )

        augmented_text, change_logs = aug.augment('The quick brown fox')[0]

        self.assertEqual(augmented_text, 'The brown fox')
        self.assertGreaterEqual(len(change_logs), 1)

    def test_keyboard_include_detail_constructor_argument(self):
        aug = nac.KeyboardAug(
            aug_char_p=1.0, aug_word_p=1.0, include_detail=True
        )

        augmented_text, change_logs = aug.augment('keyboard')[0]

        self.assertNotEqual(augmented_text, 'keyboard')
        self.assertGreaterEqual(len(change_logs), 1)
