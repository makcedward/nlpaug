import re, os
import numpy as np
from random import randint

from nlpaug.augmenter.word import WordEmbsAugmenter
from nlpaug.util import Action
import nlpaug.model.word_embs as nmw

FASTTEXT_MODEL = {}


def init_fasttext_model(model_path, force_reload=False):
    """
        Load model once at runtime
    """
    global FASTTEXT_MODEL
    if FASTTEXT_MODEL and not force_reload:
        return FASTTEXT_MODEL

    fasttext = nmw.Fasttext()
    fasttext.read(model_path)
    FASTTEXT_MODEL = fasttext

    return FASTTEXT_MODEL


class FasttextAug(WordEmbsAugmenter):
    def __init__(self, model_path='.', action=Action.SUBSTITUTE,
                 name='Fasttext_Aug', aug_min=1, aug_p=0.3, aug_n=5, tokenizer=None):
        super(FasttextAug, self).__init__(
            model_path=model_path, aug_n=aug_n,
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, tokenizer=tokenizer)
        self.model = self.get_model(force_reload=False)

    def get_model(self, force_reload=False):
        return init_fasttext_model(self.model_path, force_reload)
