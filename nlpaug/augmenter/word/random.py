"""
    Augmenter that apply random word operation to textual input.
"""

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action, Doc


class RandomWordAug(WordAugmenter):
    """
    Augmenter that apply randomly behavior for augmentation.

    :param str action: 'substitute', 'swap', 'delete' or 'crop'. If value is 'swap', adjacent words will be swapped randomly.
        If value is 'delete', word will be removed randomly. If value is 'crop', a set of contunous word will be removed randomly.
    :param float aug_p: Percentage of word will be augmented. 
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result from
        aug_p. Otherwise, using aug_max.
    :param list stopwords: List of words which will be skipped from augment operation. Not effective if action is 'crop'
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation. Not effective if action is 'crop'
    :param list target_words: List of word for replacement (used for substitute operation only). Default value is _.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.RandomWordAug()
    """

    def __init__(self, action=Action.DELETE, name='RandomWord_Aug', aug_min=1, aug_max=10, aug_p=0.3, stopwords=None,
                 target_words=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, 
                 verbose=0):
        super().__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, device='cpu', verbose=verbose,
            stopwords_regex=stopwords_regex, include_detail=False)

        self.target_words = target_words or ['_']

    # https://arxiv.org/pdf/1711.02173.pdf, https://arxiv.org/pdf/1809.02079.pdf, https://arxiv.org/pdf/1903.09460.pdf
    def swap(self, data):
        if not data or not data.strip():
            return data

        change_seq = 0
        doc = Doc(data, self.tokenizer(data))

        aug_idxes = self._get_random_aug_idxes(doc.get_original_tokens())

        # https://github.com/makcedward/nlpaug/issues/76
        if aug_idxes is None or len(aug_idxes) == 0 or doc.size() < 2:
            if self.include_detail:
                return data, []
            return data

        for aug_idx in aug_idxes:
            swap_idx = self._get_swap_position(aug_idx, doc.size() - 1)
            change_seq += 1
            doc = self.change_case(doc, aug_idx, swap_idx, change_seq)

        if self.include_detail:
            return self.reverse_tokenizer(doc.get_augmented_tokens()), doc.get_change_logs()
        else:
            return self.reverse_tokenizer(doc.get_augmented_tokens())

    # TODO: Tune it
    def change_case(self, doc, original_word_idx, swap_word_idx, change_seq):
        original_token = doc.get_token(original_word_idx).get_latest_token().token
        swap_token = doc.get_token(swap_word_idx).get_latest_token().token

        if original_word_idx != 0 and swap_word_idx != 0:
            doc.add_change_log(original_word_idx, new_token=swap_token, action=Action.SWAP,
                               change_seq=self.parent_change_seq+change_seq)
            doc.add_change_log(swap_word_idx, new_token=original_token, action=Action.SWAP,
                               change_seq=self.parent_change_seq+change_seq)
            return doc

        original_token_case = self.get_word_case(original_token)
        swap_token_case = self.get_word_case(swap_token)

        if original_word_idx == 0:
            if original_token_case == 'capitalize' and swap_token_case == 'lower':
                doc.add_change_log(original_word_idx, new_token=swap_token.capitalize(),
                                   action=Action.SWAP, change_seq=self.parent_change_seq+change_seq)
            else:
                doc.add_change_log(original_word_idx, new_token=swap_token,
                                   action=Action.SWAP, change_seq=self.parent_change_seq+change_seq)
            if original_token_case == 'capitalize':
                doc.add_change_log(swap_word_idx, new_token=original_token.lower(),
                                   action=Action.SWAP, change_seq=self.parent_change_seq+change_seq)
            else:
                doc.add_change_log(swap_word_idx, new_token=original_token,
                                   action=Action.SWAP, change_seq=self.parent_change_seq+change_seq)

        if swap_word_idx == 0:
            if original_token_case == 'lower':
                doc.add_change_log(swap_word_idx, new_token=original_token.capitalize(),
                                   action=Action.SWAP, change_seq=self.parent_change_seq+change_seq)
            else:
                doc.add_change_log(swap_word_idx, new_token=original_token,
                                   action=Action.SWAP, change_seq=self.parent_change_seq+change_seq)

            if swap_token_case == 'capitalize':
                doc.add_change_log(original_word_idx, new_token=swap_token.lower(),
                                   action=Action.SWAP, change_seq=self.parent_change_seq+change_seq)
            else:
                doc.add_change_log(original_word_idx, new_token=swap_token,
                                   action=Action.SWAP, change_seq=self.parent_change_seq+change_seq)

        # Special case for i
        if doc.get_token(original_word_idx).get_latest_token().token == 'i':
            doc.update_change_log(original_word_idx, token='I')
        if doc.get_token(swap_word_idx).get_latest_token().token == 'i':
            doc.update_change_log(swap_word_idx, token='I')

        return doc

    def _get_swap_position(self, pos, token_length):
        if pos == 0:
            # Force swap with next character if it is first character
            return pos + 1
        elif pos == token_length:
            # Force swap with previous character if it is last character
            return pos - 1
        else:
            return pos + self.sample([-1, 1], 1)[0]

    # https://arxiv.org/pdf/1703.02573.pdf, https://arxiv.org/pdf/1712.06751.pdf, https://arxiv.org/pdf/1806.09030.pdf
    # https://arxiv.org/pdf/1905.11268.pdf,
    def substitute(self, data):
        if not data or not data.strip():
            return data

        change_seq = 0
        doc = Doc(data, self.tokenizer(data))

        aug_idxes = self._get_random_aug_idxes(doc.get_original_tokens())
        aug_idxes.sort(reverse=True)

        if aug_idxes is None or len(aug_idxes) == 0:
            if self.include_detail:
                return data, []
            return data

        for aug_idx in aug_idxes:
            original_token = doc.get_token(aug_idx).orig_token.token
            new_token = self.sample(self.target_words, 1)[0]
            if aug_idx == 0:
                new_token = self.align_capitalization(original_token, new_token)

            change_seq += 1
            doc.add_change_log(aug_idx, new_token=new_token, action=Action.SUBSTITUTE, change_seq=self.parent_change_seq+change_seq)

        if self.include_detail:
            return self.reverse_tokenizer(doc.get_augmented_tokens()), doc.get_change_logs()
        else:
            return self.reverse_tokenizer(doc.get_augmented_tokens())

    # https://arxiv.org/pdf/1905.11268.pdf, https://arxiv.org/pdf/1809.02079.pdf, https://arxiv.org/pdf/1903.09460.pdf
    def delete(self, data):
        if not data or not data.strip():
            return data
            
        change_seq = 0
        doc = Doc(data, self.tokenizer(data))

        aug_idxes = self._get_random_aug_idxes(doc.get_original_tokens())
        aug_idxes.sort(reverse=True)

        # https://github.com/makcedward/nlpaug/issues/76
        if aug_idxes is None or len(aug_idxes) == 0 or doc.size() < 2:
            if self.include_detail:
                return data, []
            return data

        for aug_idx in aug_idxes:
            original_token = doc.get_token(aug_idx).orig_token.token

            change_seq += 1
            doc.add_change_log(aug_idx, new_token='', action=Action.DELETE, change_seq=self.parent_change_seq+change_seq)
            if aug_idx == 0:
                new_token = self.align_capitalization(original_token, doc.get_token(1).orig_token.token)
                doc.add_change_log(1, new_token=new_token, action=Action.ALIGN, change_seq=self.parent_change_seq+change_seq)

        if self.include_detail:
            return self.reverse_tokenizer(doc.get_augmented_tokens()), doc.get_change_logs()
        else:
            return self.reverse_tokenizer(doc.get_augmented_tokens())

    # https://github.com/makcedward/nlpaug/issues/126
    def crop(self, data):
        if not data or not data.strip():
            return data

        change_seq = 1
        doc = Doc(data, self.tokenizer(data))

        aug_idxes = self._get_aug_range_idxes(doc.get_original_tokens())
        aug_idxes.sort(reverse=True)

        # https://github.com/makcedward/nlpaug/issues/76
        if aug_idxes is None or len(aug_idxes) == 0 or doc.size() < 2:
            if self.include_detail:
                return data, []
            return data

        for aug_idx in aug_idxes:
            original_token = doc.get_token(aug_idx).orig_token.token

            doc.add_change_log(aug_idx, new_token='', action=Action.CROP, change_seq=self.parent_change_seq+change_seq)
            if aug_idx == 0:
                new_token = self.align_capitalization(original_token, doc.get_token(1).orig_token.token)
                doc.add_change_log(1, new_token=new_token, action=Action.ALIGN, change_seq=self.parent_change_seq+change_seq)

        if self.include_detail:
            return self.reverse_tokenizer(doc.get_augmented_tokens()), doc.get_change_logs()
        else:
            return self.reverse_tokenizer(doc.get_augmented_tokens())
