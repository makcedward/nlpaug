import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from nlpaug.model.lang_models.language_models import LanguageModels


class FakeTokenizer:
    def __call__(self, texts, padding=True, truncation=False, return_tensors='pt'):
        return {
            'input_ids': torch.tensor([[1, 2], [3, 4]]),
            'attention_mask': torch.tensor([[1, 1], [1, 1]]),
        }

    def convert_tokens_to_ids(self, token):
        return {'tok': 7}[token]

    def convert_ids_to_tokens(self, token_id):
        mapping = {
            0: '',
            1: '[UNK]',
            2: '##',
            3: 'unused42',
            4: '.',
            5: 'Target',
            6: 'skipme',
            7: 'valid',
            8: 'extra',
        }
        return mapping[token_id]


class FakeLanguageModel(LanguageModels):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = FakeTokenizer()
        self.model = SimpleNamespace(device=self.device, to=lambda device: setattr(self, '_moved_to', device))

    def id2token(self, _id):
        return self.tokenizer.convert_ids_to_tokens(_id)

    def is_skip_candidate(self, candidate):
        return candidate == 'skipme'


def test_language_model_basic_helpers():
    model = FakeLanguageModel(device='cpu', model_type='bert', temperature=0.5, top_k=5, top_p=0.9)

    assert model.get_default_optimize_config() == {'external_memory': 1024, 'return_proba': False}
    assert model.init_optimize({'return_proba': True}) == {'external_memory': 1024, 'return_proba': True}
    assert model.convert_device('cpu') == -1
    assert model.convert_device(None) == -1
    assert model.convert_device('cuda') == 0
    assert model.convert_device('cuda:2') == 2
    assert model.convert_device('mps') == -2
    assert model.control_randomness(torch.tensor([2.0]), {'temperature': 2.0}).item() == 1.0
    assert model.control_randomness(torch.tensor([2.0]), {'temperature': None}).item() == 2.0
    assert model.get_start_token() == '[CLS]'
    assert model.get_separator_token() == '[SEP]'
    assert model.get_mask_token() == '[MASK]'
    assert model.get_pad_token() == '[PAD]'
    assert model.get_unknown_token() == '[UNK]'
    assert model.get_subword_prefix() == '##'
    assert model.token2id('tok') == 7
    assert model.id2token(7) == 'valid'
    assert model.clean(' hello ') == 'hello'


def test_language_model_encode_and_device_helpers():
    model = FakeLanguageModel(device='cpu', model_type='roberta')
    batch = model._encode_batch(['a', 'b'])
    assert set(batch.keys()) == {'input_ids', 'attention_mask'}

    moved = model._batch_to_device(batch)
    assert moved['input_ids'].device.type == 'cpu'

    model.to('cpu')
    assert model._moved_to == 'cpu'
    assert model.get_device() == 'cpu'


def test_language_model_optional_silence_and_logits():
    logger = logging.getLogger('transformers.modeling_utils')
    original_level = logger.getEffectiveLevel()

    result = LanguageModels._load_with_optional_silence(lambda: 'ok', silence=True)
    assert result == 'ok'
    assert logger.getEffectiveLevel() == original_level

    assert LanguageModels._load_with_optional_silence(lambda: 'ok', silence=False) == 'ok'
    assert LanguageModels._model_logits(SimpleNamespace(logits='value')) == 'value'
    assert LanguageModels._model_logits(('first', 'second')) == 'first'


def test_language_model_get_candidates_filters_invalid_entries():
    model = FakeLanguageModel(device='cpu', model_type='bert')

    results = model.get_candidates(
        candidate_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        candidate_probas=None,
        target_word='target',
        n=2,
        include_punctuation=False,
    )

    assert results == [('valid', 0), ('extra', 0)]


def test_language_model_pick_uses_multinomial_candidates():
    model = FakeLanguageModel(device='cpu', model_type='bert')

    with patch.object(model, 'prob_multinomial', return_value=([7, 8], [0.8, 0.2])):
        results = model.pick(torch.tensor([1.0, 2.0]), idxes=list(range(10)), target_word=None, n=2)

    assert results == [('valid', 0.8), ('extra', 0.2)]


def test_language_model_prob_multinomial_and_filtering():
    model = FakeLanguageModel(device='cpu', model_type='bert', optimize={'return_proba': True})

    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    candidate_ids, candidate_probas = model.prob_multinomial(logits, n=2)
    assert len(candidate_ids) == 2
    assert len(candidate_probas) == 2

    filtered_logits, idxes = model.filtering(torch.tensor([1.0, 2.0, 3.0, 4.0]), {'top_k': 2, 'top_p': None})
    assert len(idxes) == 2
    assert filtered_logits.shape[0] == 2

    filtered_logits_top_p, idxes_top_p = model.filtering(torch.tensor([1.0, 2.0, 3.0, 4.0]), {'top_k': None, 'top_p': 0.8})
    assert len(idxes_top_p) >= 1
    assert filtered_logits_top_p.shape[0] == len(idxes_top_p)
