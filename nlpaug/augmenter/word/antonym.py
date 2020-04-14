"""
    Augmenter that apply semantic meaning based to textual input.
"""

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action, Doc, PartOfSpeech, WarningException, WarningName, WarningCode, WarningMessage
import nlpaug.model.word_dict as nmw


class AntonymAug(WordAugmenter):
    # https://arxiv.org/pdf/1809.02079.pdf
    """
    Augmenter that leverage semantic meaning to substitute word.

    :param str lang: Language of your text. Default value is 'eng'.
    :param float aug_p: Percentage of word will be augmented.
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result from
        aug_p. Otherwise, using aug_max.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param bool include_detail: Change detail will be returned if it is True.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.AntonymAug()
    """

    def __init__(self, name='Antonym_Aug', aug_min=1, aug_max=10, aug_p=0.3, lang='eng',
                 stopwords=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, include_detail=False,
                 verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, device='cpu', verbose=verbose,
            stopwords_regex=stopwords_regex, include_detail=include_detail)

        self.aug_src = 'wordnet'  # TODO: other source
        self.lang = lang
        self.model = self.get_model(self.aug_src, lang)

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            # Some word does not come with synonym/ antony. It will be excluded in lucky draw.
            if tokens[token_idx][1] not in ['DT']:
                results.append(token_idx)

        return results

    def _get_aug_idxes(self, tokens):
        aug_cnt = self.generate_aug_cnt(len(tokens))
        word_idxes = self.pre_skip_aug(tokens, tuple_idx=0)
        word_idxes = self.skip_aug(word_idxes, tokens)
        if len(word_idxes) == 0:
            if self.verbose > 0:
                exception = WarningException(name=WarningName.OUT_OF_VOCABULARY,
                                             code=WarningCode.WARNING_CODE_002, msg=WarningMessage.NO_WORD)
                exception.output()
            return None
        if len(word_idxes) < aug_cnt:
            aug_cnt = len(word_idxes)
        aug_idexes = self.sample(word_idxes, aug_cnt)
        return aug_idexes

    def substitute(self, data):
        change_seq = 0
        doc = Doc(data, self.tokenizer(data))

        pos = self.model.pos_tag(doc.get_original_tokens())

        aug_idxes = self._get_aug_idxes(pos)
        if aug_idxes is None or len(aug_idxes) == 0:
            if self.include_detail:
                return data, []
            return data

        for aug_idx, original_token in enumerate(doc.get_original_tokens()):
            # Skip if no augment for word
            if aug_idx not in aug_idxes:
                continue

            word_poses = PartOfSpeech.constituent2pos(pos[aug_idx][1])
            candidates = []
            if word_poses is None or len(word_poses) == 0:
                # Use every possible words as the mapping does not defined correctly
                candidates.extend(self.model.predict(pos[aug_idx][0]))
            else:
                for word_pos in word_poses:
                    candidates.extend(self.model.predict(pos[aug_idx][0], pos=word_pos))

            candidates = [c for c in candidates if c.lower() != original_token.lower()]

            if len(candidates) > 0:
                candidate = self.sample(candidates, 1)[0]
                candidate = candidate.replace("_", " ").replace("-", " ").lower()
                substitute_token = self.align_capitalization(original_token, candidate)

                if aug_idx == 0:
                    substitute_token = self.align_capitalization(original_token, substitute_token)

                change_seq += 1
                doc.add_change_log(aug_idx, new_token=substitute_token, action=Action.SUBSTITUTE,
                                   change_seq=self.parent_change_seq + change_seq)

        if self.include_detail:
            return self.reverse_tokenizer(doc.get_augmented_tokens()), doc.get_change_logs()
        else:
            return self.reverse_tokenizer(doc.get_augmented_tokens())

    @classmethod
    def get_model(cls, aug_src, lang):
        if aug_src == 'wordnet':
            return nmw.WordNet(lang=lang, is_synonym=False)
