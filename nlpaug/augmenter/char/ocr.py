"""
    Augmenter that apply ocr error simulation to textual input.
"""
import os

from nlpaug.augmenter.char import CharAugmenter
from nlpaug.util import Action, Method, Doc, ReadUtil, LibraryUtil
import nlpaug.model.char as nmc


class OcrAug(CharAugmenter):
    """
    Augmenter that simulate ocr error by random values. For example, OCR may recognize I as 1 incorrectly.\
        Pre-defined OCR mapping is leveraged to replace character by possible OCR error.

    :param float aug_char_p: Percentage of character (per token) will be augmented.
    :param int aug_char_min: Minimum number of character will be augmented.
    :param int aug_char_max: Maximum number of character will be augmented. If None is passed, number of augmentation is
        calculated via aup_char_p. If calculated result from aug_char_p is smaller than aug_char_max, will use calculated result
        from aup_char_p. Otherwise, using aug_max.
    :param float aug_word_p: Percentage of word will be augmented.
    :param int aug_word_min: Minimum number of word will be augmented.
    :param int aug_word_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_word_p. If calculated result from aug_word_p is smaller than aug_word_max, will use calculated result
        from aug_word_p. Otherwise, using aug_max.
    :param int min_char: If word less than this value, do not draw word for augmentation
    :param list stopwords: List of words which will be skipped from augment operation.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param obj dict_of_path: Use pre-defined dictionary by default. Pass either file path of dict to use custom mapping. 
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.char as nac
    >>> aug = nac.OcrAug()
    """

    def __init__(self, name='OCR_Aug', aug_char_min=2, aug_char_max=10, aug_char_p=0.3,
                 aug_word_p=0.3, aug_word_min=1, aug_word_max=10, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, verbose=0, stopwords_regex=None, min_char=1, dict_of_path=None):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, min_char=min_char, aug_char_min=aug_char_min, aug_char_max=aug_char_max,
            aug_char_p=aug_char_p, aug_word_min=aug_word_min, aug_word_max=aug_word_max, aug_word_p=aug_word_p,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu',
            verbose=verbose, stopwords_regex=stopwords_regex, include_special_char=True, include_detail=False)

        self.model = self.get_model(dict_of_path)

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            # Some character mapping do not exist. It will be excluded in lucky draw.
            char = tokens[token_idx]
            if char in self.model.model and len(self.model.predict(char)) > 0:
                results.append(token_idx)

        return results

    def substitute(self, data):
        if not data or not data.strip():
            return data
            
        change_seq = 0

        doc = Doc(data, self.tokenizer(data))
        aug_word_idxes = self._get_aug_idxes(
            doc.get_original_tokens(), self.aug_word_min, self.aug_word_max, self.aug_word_p, Method.WORD)

        for token_i, token in enumerate(doc.get_original_tokens()):
            if token_i not in aug_word_idxes:
                continue

            substitute_token = ''
            chars = self.token2char(token)
            aug_char_idxes = self._get_aug_idxes(chars, self.aug_char_min, self.aug_char_max, self.aug_char_p,
                                                 Method.CHAR)
            if aug_char_idxes is None:
                continue

            for char_i, char in enumerate(chars):
                if char_i not in aug_char_idxes:
                    substitute_token += char
                    continue

                substitute_token += self.sample(self.model.predict(chars[char_i]), 1)[0]

            # No capitalization alignment as this augmenter try to OCR engine error

            change_seq += 1
            doc.add_change_log(token_i, new_token=substitute_token, action=Action.SUBSTITUTE,
                               change_seq=self.parent_change_seq+change_seq)

        if self.include_detail:
            return self.reverse_tokenizer(doc.get_augmented_tokens()), doc.get_change_logs()
        else:
            return self.reverse_tokenizer(doc.get_augmented_tokens())

    @classmethod
    def get_model(cls, dict_of_path):
        # Use default
        if not dict_of_path:
            default_path = os.path.join(LibraryUtil.get_res_dir(), 'char', 'ocr', 'en.json')
            model = ReadUtil.read_json(default_path)
            return nmc.Ocr(model=model)

        # Use dict
        if type(dict_of_path) is dict:
            return nmc.Ocr(model=dict_of_path)

        # Use json from file
        model = ReadUtil.read_json(dict_of_path)
        if not model:
            raise ValueError('The dict_of_path does not exist. Please check "{}"'.format(dict_of_path))
        return nmc.Ocr(model=model)
