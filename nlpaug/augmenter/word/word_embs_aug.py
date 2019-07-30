import re, os
import numpy as np
from random import randint

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action


class WordEmbsAugmenter(WordAugmenter):
    def __init__(self, model_path='.', action=Action.SUBSTITUTE,
                 name='WordEmbs_Aug', aug_min=1, aug_p=0.3, aug_n=5, n_gram_separator='_',
                 stopwords=[], tokenizer=None, reverse_tokenizer=None, verbose=0):
        super().__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, verbose=verbose)
        self.model_path = model_path
        self.aug_n = aug_n
        self.model = self.get_model(force_reload=False)
        self.n_gram_separator = n_gram_separator

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            """
                Some word does not come with vector. It will be excluded in lucky draw. 
            """
            word = tokens[token_idx]

            if word in self.model.w2v:
                results.append(token_idx)

        return results

    def insert(self, text):

        tokens = self.tokenizer(text)
        results = tokens.copy()

        aug_idexes = self._get_random_aug_idxes(tokens)
        if aug_idexes is None:
            return text
        aug_idexes.sort(reverse=True)

        for aug_idx in aug_idexes:
            new_word = self.sample(self.model.get_vocab(), 1)[0]
            if self.n_gram_separator in new_word:
                new_word = new_word.split(self.n_gram_separator)[0]
            results.insert(aug_idx, new_word)

        return self.reverse_tokenizer(results)

    def substitute(self, text):
        tokens = self.tokenizer(text)
        results = tokens.copy()

        aug_idexes = self._get_aug_idxes(tokens)
        if aug_idexes is None:
            return text

        for aug_idx in aug_idexes:
            original_word = results[aug_idx]
            candidate_words = self.model.predict(original_word, top_n=self.aug_n)
            substitute_word = self.sample(candidate_words, 1)[0]

            results[aug_idx] = substitute_word

        return self.reverse_tokenizer(results)
