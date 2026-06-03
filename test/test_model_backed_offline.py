import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.word.back_translation as back_translation_module
import nlpaug.augmenter.word.context_word_embs as context_word_embs_module


class FakeTokenizer:
    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        return tokens

    def decode(self, ids):
        if isinstance(ids, list):
            return " ".join(token for token in ids if token)
        return str(ids)

    def convert_tokens_to_string(self, tokens):
        return " ".join(token for token in tokens if token)


class FakeMaskedLmModel:
    UNKNOWN_TOKEN = "[UNK]"

    def __init__(self, device="cpu", top_k=5, batch_size=32, silence=True):
        self.device = device or "cpu"
        self.top_k = top_k
        self.batch_size = batch_size
        self.silence = silence
        self.tokenizer = FakeTokenizer()
        self._model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=256))

    def get_device(self):
        return self.device

    def get_unknown_token(self):
        return self.UNKNOWN_TOKEN

    def get_subword_prefix(self):
        return "##"

    def get_mask_token(self):
        return "[MASK]"

    def get_max_num_token(self):
        return 128

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self._model

    def predict(self, masked_texts, target_words=None, n=2):
        outputs = []
        for text in masked_texts:
            if "[MASK]" not in text:
                outputs.append(["swift"])
                continue
            outputs.append(["swift", "rapid"])
        return outputs


class FakeTextGenerationModel:
    def __init__(self, device="cpu", **kwargs):
        self.device = device or "cpu"
        self.batch_size = kwargs.get("batch_size", 32)
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 0.9)
        self.silence = kwargs.get("silence", True)
        self.MASK_TOKEN = "[MASK]"
        self.tokenizer = FakeTokenizer()

    def predict(self, texts, n=1, **kwargs):
        return [[f"{text} generated." ] for text in texts] if kwargs.get("external_memory") is not None else [f"{text} generated." for text in texts]

    def get_device(self):
        return self.device


class FakeSeq2SeqModel:
    def __init__(self, device="cpu", batch_size=32, max_length=None, **kwargs):
        self.device = device or "cpu"
        self.batch_size = batch_size
        self.max_length = max_length

    def get_device(self):
        return self.device

    def predict(self, texts, n=1):
        if isinstance(texts, str):
            texts = [texts]
        return [f"{text} translated" for text in texts]


class FakeSummaryModel:
    def __init__(self, device="cpu", batch_size=32, **kwargs):
        self.device = device or "cpu"
        self.batch_size = batch_size
        self.min_length = kwargs.get("min_length", 20)
        self.max_length = kwargs.get("max_length", 50)
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 0.9)

    def get_device(self):
        return self.device

    def predict(self, texts, n=1):
        return ["summary." for _ in texts]


class FakeLambadaModel:
    def __init__(self, device="cpu", batch_size=16, threshold=None, **kwargs):
        self.device = device or "cpu"
        self.batch_size = batch_size
        self.threshold = threshold
        self.min_length = kwargs.get("min_length", 100)
        self.max_length = kwargs.get("max_length", 300)
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 0.9)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

    def get_device(self):
        return self.device

    def predict(self, labels, n=10):
        rows = []
        for idx, label in enumerate(labels):
            for _ in range(n):
                rows.append({"id": idx, "text": f"{label} sample", "label": label})
        return pd.DataFrame(rows)


def test_contextual_word_embs_offline_substitute_and_batch():
    fake_model = FakeMaskedLmModel(device="cpu")
    with patch.object(naw.ContextualWordEmbsAug, "get_model", return_value=fake_model):
        aug = naw.ContextualWordEmbsAug(model_path="bert-base-uncased", device=None, batch_size=2)
        assert aug.device == "cpu"

        text = "The quick brown fox jumps over the lazy dog"
        augmented = aug.augment(text)
        assert isinstance(augmented, list)
        assert len(augmented) == 1
        assert augmented[0] != text

        batch_augmented = aug.augment([text, text])
        assert len(batch_augmented) == 2
        assert all(item != text for item in batch_augmented)


def test_contextual_word_embs_cache_reuses_model_and_updates_runtime_attrs():
    context_word_embs_module.CONTEXT_WORD_EMBS_MODELS.clear()
    created = []

    def factory(**kwargs):
        model = FakeMaskedLmModel(
            device=kwargs["device"],
            top_k=kwargs["top_k"],
            batch_size=kwargs["batch_size"],
            silence=kwargs["silence"],
        )
        created.append(model)
        return model

    with patch("nlpaug.augmenter.word.context_word_embs._create_context_word_embs_model", side_effect=factory):
        first = context_word_embs_module.init_context_word_embs_model(
            model_path="bert-base-uncased",
            model_type="bert",
            device="cpu",
            batch_size=2,
            top_k=5,
            silence=True,
            use_custom_api=True,
        )
        second = context_word_embs_module.init_context_word_embs_model(
            model_path="bert-base-uncased",
            model_type="bert",
            device="cpu",
            batch_size=8,
            top_k=11,
            silence=False,
            use_custom_api=True,
        )

    assert first is second
    assert len(created) == 1
    assert second.batch_size == 8
    assert second.top_k == 11
    assert second.silence is False


def test_back_translation_offline_and_cache_reset():
    fake_model = FakeSeq2SeqModel(device="cpu", batch_size=4, max_length=128)
    with patch.object(naw.BackTranslationAug, "get_model", return_value=fake_model):
        aug = naw.BackTranslationAug(device="cpu", batch_size=4, max_length=128)
        result = aug.augment("The quick brown fox")
        assert result == ["The quick brown fox translated"]

        batch = aug.augment(["a", "b"])
        assert batch == ["a translated", "b translated"]

        naw.BackTranslationAug.clear_cache()


def test_back_translation_cache_reuses_model_and_updates_runtime_attrs():
    back_translation_module.BACK_TRANSLATION_MODELS.clear()
    created = []

    def factory(**kwargs):
        model = FakeSeq2SeqModel(
            device=kwargs["device"],
            batch_size=kwargs["batch_size"],
            max_length=kwargs["max_length"],
        )
        created.append(model)
        return model

    with patch("nlpaug.augmenter.word.back_translation.nml.MtTransformers", side_effect=factory):
        first = back_translation_module.init_back_translation_model(
            from_model_name="from",
            to_model_name="to",
            device="cpu",
            batch_size=2,
            max_length=32,
        )
        second = back_translation_module.init_back_translation_model(
            from_model_name="from",
            to_model_name="to",
            device="cpu",
            batch_size=7,
            max_length=64,
        )

    assert first is second
    assert len(created) == 1
    assert second.batch_size == 7
    assert second.max_length == 64


def test_abstractive_summarization_offline():
    fake_model = FakeSummaryModel(device="cpu", batch_size=2)
    with patch.object(nas.AbstSummAug, "get_model", return_value=fake_model):
        aug = nas.AbstSummAug(model_path="t5-small", device="cpu", batch_size=2)
        result = aug.augment(["One long sentence here.", "Another long sentence here."])
        assert result == ["summary.", "summary."]


def test_sentence_contextual_generation_offline():
    fake_model = FakeTextGenerationModel(device="cpu", batch_size=2)
    with patch.object(nas.ContextualWordEmbsForSentenceAug, "get_model", return_value=fake_model):
        aug = nas.ContextualWordEmbsForSentenceAug(model_path="distilgpt2", device=None, batch_size=2, use_custom_api=False)
        assert aug.device == "cpu"

        text = "The quick brown fox"
        result = aug.augment(text)
        assert len(result) == 1
        assert result[0].startswith(text)

        batch = aug.augment([text, text])
        assert len(batch) == 2


def test_lambada_offline(tmp_path):
    model_dir = Path(tmp_path) / "lambada"
    cls_dir = model_dir / "cls"
    gen_dir = model_dir / "gen"
    cls_dir.mkdir(parents=True)
    gen_dir.mkdir(parents=True)
    (cls_dir / "label_encoder.json").write_text(json.dumps({"0": 0, "1": 1}), encoding="utf8")

    fake_model = FakeLambadaModel(device="cpu", batch_size=2, threshold=None)
    with patch.object(nas.LambadaAug, "get_model", return_value=fake_model):
        aug = nas.LambadaAug(model_dir=str(model_dir), device="cpu", threshold=None, batch_size=2)
        result = aug.augment(["0", "1"], n=2)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4

        try:
            aug.augment(["missing"], n=1)
            assert False, "Expected invalid label to raise"
        except Exception as exc:
            assert "does not exist" in str(exc)
