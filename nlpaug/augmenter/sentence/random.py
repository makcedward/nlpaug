"""
    Augmenter that apply operation (sentence level) to textual input based on abstractive summarization.
"""

import os


from nlpaug.augmenter.sentence import SentenceAugmenter
import nlpaug.model.word_rule as nmr
from nlpaug.util import Action, Doc


class RandomSentAug(SentenceAugmenter):

    """
    Augmenter that apply randomly behavior for augmentation.

    :param str mode: Shuffle sentence to left, right, neighbor or random position. For `left`, target sentence
        will be swapped with left sentnece. For `right`, target sentence will be swapped with right sentnece.
        For `neighbor`, target sentence will be swapped with left or right sentnece radomly. For `random`, 
        target sentence will be swapped with any sentnece randomly.
    :param float aug_p: Percentage of sentence will be augmented. 
    :param int aug_min: Minimum number of sentence will be augmented.
    :param int aug_max: Maximum number of sentence will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result from
        aug_p. Otherwise, using aug_max.
    :param func tokenizer: Customize tokenization process
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.sentence as nas
    >>> aug = nas.RandomSentAug()
    """

    def __init__(self, mode='neighbor', action=Action.SWAP, name='RandomSent_Aug', aug_min=1, aug_max=10, aug_p=0.3,
        tokenizer=None, verbose=0):
        super().__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, verbose=verbose)

        self.model = nmr.Shuffle(mode=mode, model_type='sentence', tokenizer=tokenizer)

    def pre_skip_aug(self, data):
        return list(range(len(data)))
        
    # https://arxiv.org/abs/1910.13461
    def swap(self, data):
        if not data:
            return data

        if isinstance(data, list):
            all_data = data
        else:
            if data.strip() == '':
                return data
            all_data = [data]

        for i, d in enumerate(all_data):
            sentences = self.model.tokenize(d)
            aug_idxes = self._get_random_aug_idxes(sentences)
            for aug_idx in aug_idxes:
                sentences = self.model.predict(sentences, aug_idx)
            all_data[i] = ' '.join(sentences)

        # TODO: always return array
        if isinstance(data, list):
            return all_data
        else:
            return all_data[0]

