import logging

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels


class MtTransformers(LanguageModels):
    def __init__(self, src_model_name='Helsinki-NLP/opus-mt-en-jap', tgt_model_name='Helsinki-NLP/opus-mt-jap-en', 
        device='cuda', silence=True):
        super().__init__(device, model_type=None, silence=silence)
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed transformers library. Install transfomers by `pip install transformers`')

        self.src_model_name = src_model_name
        self.tgt_model_name = tgt_model_name
        self.src_model = AutoModelForSeq2SeqLM.from_pretrained(self.src_model_name)
        self.tgt_model = AutoModelForSeq2SeqLM.from_pretrained(self.tgt_model_name)
        self.src_tokenizer = AutoTokenizer.from_pretrained(self.src_model_name)
        self.tgt_tokenizer = AutoTokenizer.from_pretrained(self.tgt_model_name)

    def predict(self, texts, target_words=None, n=1):
        src_tokenized_texts = self.src_tokenizer.prepare_seq2seq_batch(texts, return_tensors='pt')
        src_translated_ids = self.src_model.generate(**src_tokenized_texts)
        src_translated_texts = self.src_tokenizer.batch_decode(src_translated_ids, skip_special_tokens=True)

        tgt_tokenized_texts = self.tgt_tokenizer.prepare_seq2seq_batch(src_translated_texts, return_tensors='pt')
        tgt_translated_ids = self.tgt_model.generate(**tgt_tokenized_texts)
        tgt_translated_texts = self.tgt_tokenizer.batch_decode(tgt_translated_ids, skip_special_tokens=True)

        return tgt_translated_texts
