import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nlpaug.augmenter.word import context_word_embs as cwe_mod
from nlpaug.augmenter.word.context_word_embs import ContextualWordEmbsAug
from nlpaug.model.lang_models.language_models import LanguageModels


def make_aug_without_init(model_path):
    aug = object.__new__(ContextualWordEmbsAug)
    aug.model_path = model_path
    return aug


class FakeModel(LanguageModels):
    def __init__(self, model_type):
        # Skip torch-dependent __init__; only model_type matters here.
        self.device = 'cpu'
        self.model_type = model_type


class TestXlmRobertaModelTypeDetection(unittest.TestCase):
    def test_detects_xlm_roberta_before_roberta(self):
        self.assertEqual(
            'xlm-roberta',
            make_aug_without_init('xlm-roberta-base').check_model_type())
        self.assertEqual(
            'xlm-roberta',
            make_aug_without_init('FacebookAI/xlm-roberta-large').check_model_type())

    def test_existing_detections_unchanged(self):
        cases = {
            'bert-base-uncased': 'bert',
            'bert-base-cased': 'bert',
            'distilbert-base-uncased': 'bert',
            'squeezebert/squeezebert-uncased': 'bert',
            'roberta-base': 'roberta',
            'distilroberta-base': 'roberta',
            'allenai/longformer-base-4096': 'roberta',
            'facebook/bart-base': 'bart',
            'unknown-model-xyz': '',
        }
        for path, expected in cases.items():
            self.assertEqual(expected, make_aug_without_init(path).check_model_type())


class TestXlmRobertaRouting(unittest.TestCase):
    def test_uses_fill_mask_pipeline_not_custom_api(self):
        # The custom Roberta backend hardcodes the Ġ prefix, which is wrong
        # for SentencePiece-based XLM-R. Both paths must use FmTransformers.
        for use_custom_api in (True, False):
            with patch.object(
                    cwe_mod.nml, 'FmTransformers', return_value=SimpleNamespace()) as fm:
                cwe_mod._create_context_word_embs_model(
                    'xlm-roberta-base', 'xlm-roberta', device='cpu',
                    batch_size=32, top_k=5, silence=True,
                    use_custom_api=use_custom_api)
                fm.assert_called_once()
                self.assertEqual('xlm-roberta', fm.call_args[1]['model_type'])

    def test_custom_api_still_used_for_plain_roberta(self):
        with patch.object(
                cwe_mod.nml, 'Roberta', return_value=SimpleNamespace()) as roberta:
            cwe_mod._create_context_word_embs_model(
                'roberta-base', 'roberta', device='cpu',
                batch_size=32, top_k=5, silence=True, use_custom_api=True)
            roberta.assert_called_once()


class TestXlmRobertaTokens(unittest.TestCase):
    def test_mask_pad_unk_like_roberta(self):
        model = FakeModel('xlm-roberta')
        self.assertEqual('<mask>', model.get_mask_token())
        self.assertEqual('<pad>', model.get_pad_token())
        self.assertEqual('<unk>', model.get_unknown_token())
        self.assertEqual('<s>', model.get_start_token())
        self.assertEqual('</s>', model.get_separator_token())

    def test_subword_prefix_is_sentencepiece(self):
        self.assertEqual('▁', FakeModel('xlm-roberta').get_subword_prefix())
        # Plain RoBERTa keeps its byte-level prefix.
        self.assertEqual('Ġ', FakeModel('roberta').get_subword_prefix())


if __name__ == '__main__':
    unittest.main()
