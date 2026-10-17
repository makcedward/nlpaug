import inspect
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from nlpaug.model.lang_models.language_models import LanguageModels
from nlpaug.model.lang_models.lambada import Lambada
from nlpaug.model.lang_models.machine_translation_transformers import MtTransformers


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


def test_language_model_normalizes_torch_device_values():
    model = FakeLanguageModel(device=torch.device('cpu'), model_type='bert')

    assert model.device == 'cpu'
    assert model.convert_device(torch.device('cpu')) == -1
    assert model.convert_device(torch.device('cuda', 2)) == 2

    filtered_logits, idxes = model.filtering(
        torch.tensor([1.0, 2.0, 3.0]),
        {'top_k': 2, 'top_p': None},
    )
    assert filtered_logits.shape[0] == 2
    assert len(idxes) == 2


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


def test_lambada_generate_uses_supported_transformers_arguments():
    class FakeGenerationModel:
        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return torch.tensor([[1, 2, 3]])

    class FakeGenerationTokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False, return_tensors='pt'):
            return torch.tensor([[1, 2]])

        @staticmethod
        def decode(sequence, clean_up_tokenization_spaces=True):
            return '0 [SEP] generated<|endoftext|>'

    model = object.__new__(Lambada)
    model.device = 'cpu'
    model.batch_size = 2
    model.min_length = 1
    model.max_length = 10
    model.temperature = 1.0
    model.top_k = 50
    model.top_p = 0.9
    model.repetition_penalty = 1.0
    model.sep_token = '[SEP]'
    model.stop_token = '<|endoftext|>'
    model.gen_model = FakeGenerationModel()
    model.gen_tokenizer = FakeGenerationTokenizer()

    result = model._generate(['0'], n=1)

    assert model.gen_model.kwargs['num_return_sequences'] == 1
    assert 'num_workers' not in model.gen_model.kwargs
    assert result.to_dict('records') == [{'id': 0, 'text': ' generated', 'label': '0'}]


def test_translation_forwards_all_tokenizer_fields_in_named_batches():
    class FakeTranslationTokenizer:
        def __call__(self, data, **kwargs):
            assert kwargs == {'padding': True, 'truncation': True, 'max_length': 32,
                              'return_tensors': 'pt'}
            return {
                'input_ids': torch.tensor([[1, 2], [3, 4], [5, 6]]),
                'attention_mask': torch.ones((3, 2), dtype=torch.long),
                'token_type_ids': torch.zeros((3, 2), dtype=torch.long),
            }

        @staticmethod
        def batch_decode(batch, skip_special_tokens=True):
            assert skip_special_tokens is True
            return [f'translated-{int(row[0])}' for row in batch]

    class FakeTranslationModel:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return kwargs['input_ids']

    translator = object.__new__(MtTransformers)
    translator.device = 'cpu'
    translator.batch_size = 2
    translator.max_length = 32
    tokenizer = FakeTranslationTokenizer()
    model = FakeTranslationModel()

    result = translator.translate_one_step_batched(['a', 'b', 'c'], tokenizer, model)

    assert result == ['translated-1', 'translated-3', 'translated-5']
    assert [call['input_ids'].shape[0] for call in model.calls] == [2, 1]
    assert all('token_type_ids' in call for call in model.calls)
    assert all(call['max_length'] == 32 for call in model.calls)


def test_pipeline_augmenters_use_single_process_workers():
    # Regression test for issue #330: DataLoader workers crash on macOS.
    # HF pipelines must be called with num_workers=0 (no worker processes).
    import pathlib
    repo = pathlib.Path(__file__).resolve().parents[3] / 'nlpaug' / 'model' / 'lang_models'
    for fname in ('fill_mask_transformers.py', 'text_generation_transformers.py',
                  'summarization_transformers.py'):
        content = (repo / fname).read_text()
        assert 'num_workers=0' in content, fname
        for bad in ('num_workers=1)', 'num_workers=1,', 'num_workers=1\n', 'num_workers = 1'):
            assert bad not in content, (fname, bad)


def test_filtering_works_without_numpy_available():
    # Regression test for issue #320: torch built without numpy raises
    # `RuntimeError: Numpy is not available` on `tensor.numpy()`. filtering()
    # must use the torch-native `.cpu().tolist()` path instead.
    model = FakeLanguageModel(device='cpu', model_type='bert')

    def _boom(self):
        raise RuntimeError('Numpy is not available')

    with patch.object(torch.Tensor, 'numpy', _boom):
        filtered_top_k, idxes_top_k = model.filtering(
            torch.tensor([1.0, 2.0, 3.0, 4.0]), {'top_k': 2, 'top_p': None})
        assert len(idxes_top_k) == 2
        assert filtered_top_k.shape[0] == 2
        assert sorted(idxes_top_k) == [2, 3]

        filtered_top_p, idxes_top_p = model.filtering(
            torch.tensor([1.0, 2.0, 3.0, 4.0]), {'top_k': None, 'top_p': 0.8})
        assert len(idxes_top_p) >= 1
        assert filtered_top_p.shape[0] == len(idxes_top_p)

    assert LanguageModels._idxes_to_list([1, 2]) == [1, 2]


def test_lambada_validates_model_dirs_issue_335(tmp_path):
    cls_dir = tmp_path / 'cls'
    gen_dir = tmp_path / 'gen'
    cls_dir.mkdir()
    gen_dir.mkdir()

    # missing label_encoder.json
    with pytest.raises(ValueError, match='label_encoder'):
        Lambada._validate_model_dirs(str(cls_dir), str(gen_dir))

    (cls_dir / 'label_encoder.json').write_text('{"0": 0}')
    # gen dir containing cls_config.json means swapped dirs
    (gen_dir / 'cls_config.json').write_text('{}')
    with pytest.raises(ValueError, match='gen_model_dir'):
        Lambada._validate_model_dirs(str(cls_dir), str(gen_dir))
    (gen_dir / 'cls_config.json').unlink()

    Lambada._validate_model_dirs(str(cls_dir), str(gen_dir))

    with pytest.raises(ValueError, match='must be different'):
        Lambada._validate_model_dirs(str(cls_dir), str(cls_dir))


def test_lambada_strip_prompt_avoids_gibberish_truncation_issue_328():
    assert Lambada._strip_prompt('label [SEP] hello', 'label [SEP]', '[SEP]') == ' hello'
    # BPE re-spacing: fall back to sep_token split instead of blind char slice
    assert Lambada._strip_prompt('label[SEP]hello', 'label [SEP]', '[SEP]') == 'hello'
    assert Lambada._strip_prompt('hello world', 'label [SEP]', '[SEP]') == 'hello world'


def test_xlnet_decode_accepts_str_issue_316():
    # Fast tokenizers raise "Can't extract `str` to `Vec`" when a str is
    # passed to convert_tokens_to_string instead of List[str].
    from nlpaug.augmenter.sentence.context_word_embs_sentence import (
        ContextualWordEmbsForSentenceAug,
    )

    class FakeFastTokenizer:
        def convert_tokens_to_string(self, tokens):
            if isinstance(tokens, str):
                raise ValueError("Can't extract `str` to `Vec`")
            return ' '.join(tokens)

    aug = ContextualWordEmbsForSentenceAug.__new__(ContextualWordEmbsForSentenceAug)
    aug.model = SimpleNamespace(tokenizer=FakeFastTokenizer())
    assert aug._decode_xlnet_text('hello world') == 'hello world'


def test_mt_max_length_clamping_issue_339():
    cfg = SimpleNamespace(
        max_length=512, max_position_embeddings=512, n_positions=None,
        max_target_positions=None, max_source_positions=None)
    model = SimpleNamespace(config=cfg, generation_config=None)
    assert MtTransformers._resolve_model_max_length(model) == 512
    assert MtTransformers._resolve_model_max_length(SimpleNamespace(config=None, generation_config=None)) == 512
    assert MtTransformers._clamp_max_length(None, 512, 512) == 512
    assert MtTransformers._clamp_max_length(100, 512, 512) == 100
    with pytest.warns(UserWarning, match='339'):
        assert MtTransformers._clamp_max_length(1024, 512, 512) == 512


def test_textgen_warns_when_tokenized_input_exceeds_max_length_issue_285():
    import warnings
    from nlpaug.model.lang_models.text_generation_transformers import TextGenTransformers

    model = TextGenTransformers.__new__(TextGenTransformers)
    model.max_length = 10
    model.model = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda t: list(range(20))))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model._warn_if_input_exceeds_max_length(['hello world'])
    assert any('285' in str(w.message) for w in caught)

    model.model = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda t: list(range(5))))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model._warn_if_input_exceeds_max_length(['hi'])
    assert len(caught) == 0


def test_back_translation_substitute_repeats_for_n_issue_288():
    from unittest.mock import MagicMock
    from nlpaug.augmenter.word.back_translation import BackTranslationAug

    aug = BackTranslationAug.__new__(BackTranslationAug)
    aug.model = MagicMock()
    aug.model.predict.side_effect = lambda d: [f'bt:{d}'] if isinstance(d, str) else [f'bt:{x}' for x in d]
    assert aug.substitute('hello', n=3) == ['bt:hello'] * 3
    assert aug.substitute(['a', 'b']) == ['bt:a', 'bt:b']


def test_causal_sentence_aug_warns_on_default_sampling_issue_346():
    import warnings
    from unittest.mock import MagicMock, patch
    from nlpaug.augmenter.sentence.context_word_embs_sentence import ContextualWordEmbsForSentenceAug

    with patch.object(ContextualWordEmbsForSentenceAug, 'get_model', return_value=MagicMock(device='cpu')):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ContextualWordEmbsForSentenceAug(model_path='gpt2', temperature=1.0, top_k=50)
        assert any('346' in str(w.message) for w in caught)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ContextualWordEmbsForSentenceAug(model_path='gpt2', temperature=0.7, top_k=40)
        assert len(caught) == 0


def test_bert_supports_from_tf_issue_292():
    from nlpaug.model.lang_models import bert as bert_mod

    assert 'from_tf' in inspect.signature(bert_mod.Bert.__init__).parameters


def test_synonym_antonym_vbz_inflection_issue_257():
    from nlpaug.augmenter.word.synonym import inflect_like_original

    assert inflect_like_original('acknowledge', 'agrees', 'VBZ') == 'acknowledges'
    assert inflect_like_original('fly', 'flies', 'VBZ') == 'flies'
    assert inflect_like_original('watch', 'watches', 'VBZ') == 'watches'
    # Other tenses/pos are left untouched (backward compatible)
    assert inflect_like_original('acknowledge', 'agreed', 'VBD') == 'acknowledge'
    assert inflect_like_original('fast', 'faster', 'JJR') == 'fast'


def test_word_embs_predict_oov_and_no_change_warning_issues_327():
    import warnings
    from unittest.mock import MagicMock
    from nlpaug.model.word_embs.word_embeddings import WordEmbeddings
    from nlpaug.augmenter.word.word_embs import WordEmbsAug

    model = WordEmbeddings(top_k=5)
    model.model = MagicMock()
    model.model.most_similar.side_effect = KeyError('oov')
    assert model.predict('oovword') == []

    aug = WordEmbsAug.__new__(WordEmbsAug)
    aug.include_detail = False
    aug.verbose = 0
    aug.parent_change_seq = 0
    aug.tokenizer = lambda x: x.split()
    aug.reverse_tokenizer = lambda toks: ' '.join(toks)
    aug._get_aug_idxes = lambda toks: []
    aug.align_capitalization = lambda o, s: s
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert aug.substitute('hello world') == 'hello world'
    assert any('327' in str(w.message) for w in caught)


def test_word_embs_rejects_path_string_model_issue_210():
    from nlpaug.augmenter.word.word_embs import WordEmbsAug

    with pytest.raises(TypeError, match='model_path'):
        WordEmbsAug(model_type='fasttext', model='train_ft.bin', action='substitute')


def test_gpt2_batch_padding_uses_own_length_issue_266():
    from nlpaug.model.lang_models.gpt2 import Gpt2

    model = Gpt2.__new__(Gpt2)
    model.tokenizer = SimpleNamespace(
        encode=lambda t: {'short': [1, 2], 'much longer text here': [1, 2, 3, 4, 5]}[t])
    model.device = 'cpu'
    model.pad_id = 1
    model.temperature = 1.0
    model.top_k = None
    model.top_p = None
    captured = {}

    class FakeModel:
        def __call__(self, input_ids=None, attention_mask=None, past_key_values=None):
            captured['input'] = tuple(input_ids.shape)
            captured['mask'] = tuple(attention_mask.shape)
            return (torch.randn(2, 1, 10),)

    model.model = FakeModel()
    model.control_randomness = lambda logits, seed: logits
    model.filtering = lambda logits, seed: (logits, [])

    assert model.predict(['short', 'much longer text here']) == [[''], ['']]
    assert captured['input'] == captured['mask'] == (2, 5)
