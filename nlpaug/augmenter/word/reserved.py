"""
    Augmenter that apply target wordr replacement operation to textual input.
"""

from collections import defaultdict

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action, Doc


class ReservedAug(WordAugmenter):
    """
    Augmenter that apply target word replacement for augmentation.

    :param float aug_p: Percentage of word will be augmented. 
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result from
        aug_p. Otherwise, using aug_max.
    :param list reserved_tokens: A list of swappable tokens (a list of list). For example, "FWD", "Fwd" and "FW" 
        are referring to "foward" in email communcation while "Sincerely" and "Best Regards" treated as same 
        meaning. The input should be [["FWD", "Fwd", "FW"], ["Sincerely", "Best Regards"]]. 
    :param bool case_sensitive: Default is True. If True, it will only replace alternative token if all cases are same.
    :param bool allow_original: Default is False. If False, it will not pick the origianl token. For example, the
        reserved_tokens is [["FWD", "Fwd", "FW"]] and the augmented word is FDW. Only "Fwd" and "FW" will be picked.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.ReservedAug()
    """

    def __init__(self, reserved_tokens, action=Action.SUBSTITUTE, case_sensitive=True, allow_original=False, 
        name='Reserved_Aug', aug_min=1, aug_max=10, aug_p=0.3, tokenizer=None, reverse_tokenizer=None, 
        verbose=0):
        super().__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, tokenizer=tokenizer, 
            reverse_tokenizer=reverse_tokenizer, device='cpu', verbose=verbose, include_detail=False)

        self.reserved_tokens = reserved_tokens        
        self.case_sensitive = case_sensitive
        self.allow_original = allow_original
        self.reserved_token_dict = defaultdict(int)

        for i, tokens in enumerate(reserved_tokens):
            for t in tokens:
                if not case_sensitive:
                    t = t.lower()

                # If duplicates word occurs, pick the last one.
                self.reserved_token_dict[t] = i

    def skip_aug(self, token_idxes, tokens):
        # https://arxiv.org/pdf/2007.02033.pdf
        results = []
        for idx in token_idxes:
            t = tokens[idx]
            if not self.case_sensitive:
                t = t.lower()

            if t in self.reserved_token_dict:
                results.append(idx)

        return results

    def substitute(self, data):
        change_seq = 0
        doc = Doc(data, self.tokenizer(data))

        aug_idxes = self._get_aug_idxes(doc.get_original_tokens())
        aug_idxes.sort(reverse=True)

        tokens = doc.get_original_tokens()

        if aug_idxes is None or len(aug_idxes) == 0:
            if self.include_detail:
                return data, []
            return data

        for aug_idx in aug_idxes:
            original_token = doc.get_token(aug_idx).orig_token.token
            candidate_tokens = self.reserved_tokens[self.reserved_token_dict[original_token]]
            if not self.allow_original:
                candidate_tokens = [t for t in candidate_tokens if t != original_token]
            new_token = self.sample(candidate_tokens, 1)[0]
            if aug_idx == 0:
                new_token = self.align_capitalization(original_token, new_token)

            change_seq += 1
            doc.add_change_log(aug_idx, new_token=new_token, action=Action.SUBSTITUTE, change_seq=self.parent_change_seq+change_seq)

        if self.include_detail:
            return self.reverse_tokenizer(doc.get_augmented_tokens()), doc.get_change_logs()
        else:
            return self.reverse_tokenizer(doc.get_augmented_tokens())

