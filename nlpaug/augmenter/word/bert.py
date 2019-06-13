import re, os
import numpy as np
from random import randint

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action
import nlpaug.model.lang_models as nml

BERT_MODEL = {}


def init_bert_model(model_path, tokenizer_path, force_reload=False):
    """
        Load model once at runtime
    """
    global BERT_MODEL
    if BERT_MODEL and not force_reload:
        return BERT_MODEL

    bert_model = nml.Bert(model_path, tokenizer_path)
    bert_model.model.eval()
    BERT_MODEL = bert_model

    return bert_model


class BertAug(WordAugmenter):
    def __init__(self, model_path='bert-base-uncased', tokenizer_path='bert-base-uncased', action=Action.SUBSTITUTE,
                 name='Bert_Aug', aug_min=1, aug_p=0.3, aug_n=5, stopwords=[], verbose=0):
        super(BertAug, self).__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, tokenizer=None, stopwords=stopwords,
            verbose=verbose)
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.aug_n = aug_n
        self.model = self.get_model(force_reload=False)
        self.tokenizer = self.model.tokenizer.tokenize

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            """
                Some token is not a partial word.
            """
            word = tokens[token_idx]

            if word[:2] != nml.Bert.SUBWORD_PREFIX:
                results.append(token_idx)

        return results

    def reverse_tokenizer(self, tokens):
        result = ''
        for token in tokens:
            if token[:2] == nml.Bert.SUBWORD_PREFIX:
                result += token[2:]
            else:
                result += ' ' + token
        return result[1:]

    def insert(self, text):
        tokens = self.tokenizer(text)
        results = tokens.copy()

        aug_idxes = self._get_aug_idxes(tokens)
        aug_idxes.sort(reverse=True)

        for aug_idx in aug_idxes:
            results.insert(aug_idx, nml.Bert.MASK)
            new_word = self.sample(self.model.predict(results, nml.Bert.MASK, self.aug_n), 1)[0]
            results[aug_idx] = new_word

        return self.reverse_tokenizer(results)

    def substitute(self, text):
        tokens = self.tokenizer(text)
        results = tokens.copy()

        aug_idxes = self._get_aug_idxes(tokens)

        for aug_idx in aug_idxes[:1]:
            original_word = results[aug_idx]
            candidate_words = self.model.predict(results, original_word, top_n=self.aug_n)
            substitute_word = self.sample(candidate_words, 1)[0]

            results[aug_idx] = substitute_word

        final_results = []
        for src, dest in zip(tokens, results):
            final_results.append(self.align_capitalization(src, dest))

        return self.reverse_tokenizer(final_results)

    def get_model(self, force_reload=False):
        return init_bert_model(self.model_path, self.tokenizer_path, force_reload)
