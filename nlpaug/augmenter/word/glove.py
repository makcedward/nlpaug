import re, os
import numpy as np
from random import randint
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action

GLOVE_MODEL = {}


def init_glove_model(model_path, force_reload=False):
    """
        Load model once at runtime
    """
    global GLOVE_MODEL
    if GLOVE_MODEL and not force_reload:
        return GLOVE_MODEL

    try:
        # Load if it is word2vec format
        GLOVE_MODEL = KeyedVectors.load_word2vec_format(datapath(model_path), binary=False)
    except Exception:
        # Otherwise, convert it to word2vec format first
        output_path = os.path.join(
            os.path.dirname(model_path), 'glove_vectors.txt')
        glove2word2vec(glove_input_file=datapath(model_path), word2vec_output_file=datapath(output_path))

        GLOVE_MODEL = KeyedVectors.load_word2vec_format(datapath(output_path), binary=False)

    return GLOVE_MODEL


class GloVeAug(WordAugmenter):
    def __init__(self, model_path='.', action=Action.SUBSTITUTE,
                 name='GloVeAug_Aug', aug_min=1, aug_p=0.3, aug_n=5, tokenizer=None):
        super(GloVeAug, self).__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, tokenizer=tokenizer)

        self.model_path = model_path
        self.aug_n = aug_n
        self.model = self.get_model(force_reload=False)

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            """
                Some word does not come with vector. It will be excluded in lucky draw. 
            """
            word = tokens[token_idx]

            if word in self.model:
                results.append(token_idx)

        return results

    def insert(self, tokens):
        results = tokens.copy()

        aug_cnt = self.generate_aug_cnt(len(tokens))
        word_idxes = [i for i, t in enumerate(tokens)]
        aug_idexes = self.sample(word_idxes, aug_cnt)
        aug_idexes.sort(reverse=True)

        for aug_idx in aug_idexes:
            new_word = self.sample(list(self.model.vocab), 1)[0]
            results.insert(aug_idx, new_word)

        return results

    def substitute(self, tokens):
        results = tokens.copy()

        aug_cnt = self.generate_aug_cnt(len(tokens))
        word_idxes = [i for i, t in enumerate(tokens)]
        aug_idexes = self.sample(word_idxes, aug_cnt)

        for aug_idx in aug_idexes:
            original_word = results[aug_idx]
            candidate_words = self.model.most_similar(original_word, topn=self.aug_n)
            substitute_word = self.sample(candidate_words, 1)[0][0]

            results[aug_idx] = substitute_word

        return results

    def get_model(self, force_reload=False):
        return init_glove_model(self.model_path, force_reload)
