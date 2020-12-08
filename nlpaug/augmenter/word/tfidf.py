"""
    Augmenter that apply TF-IDF based to textual input.
"""

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action, Doc, WarningException, WarningName, WarningCode, WarningMessage
import nlpaug.model.word_stats as nmws

TFIDF_MODEL = {}


def init_tfidf_model(model_path, force_reload=False):
    # Load model once at runtime
    global TFIDF_MODEL
    if TFIDF_MODEL and not force_reload:
        return TFIDF_MODEL

    tfidf_model = nmws.TfIdf()
    tfidf_model.read(model_path)
    TFIDF_MODEL = tfidf_model

    return tfidf_model


class TfIdfAug(WordAugmenter):
    # https://arxiv.org/pdf/1904.12848.pdf
    """
    Augmenter that leverage TF-IDF statistics to insert or substitute word.

    :param str model_path: Downloaded model directory. Either model_path or model is must be provided
    :param str action: Either 'insert or 'substitute'. If value is 'insert', a new word will be injected to random
        position according to TF-IDF calculation. If value is 'substitute', word will be replaced according
        to TF-IDF calculation
    :param int top_k: Controlling lucky draw pool. Top k score token will be used for augmentation. Larger k, more
        token can be used. Default value is 5. If value is None which means using all possible tokens.
    :param float aug_p: Percentage of word will be augmented.
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result
        from aug_p. Otherwise, using aug_max.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.TfIdfAug(model_path='.')
    """

    def __init__(self, model_path='.', action=Action.SUBSTITUTE,
                 name='TfIdf_Aug', aug_min=1, aug_max=10, aug_p=0.3, top_k=5, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, verbose=0):
        super().__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, device='cpu', verbose=verbose,
            stopwords_regex=stopwords_regex, include_detail=False)
        self.model_path = model_path
        self.top_k = top_k
        self.model = self.get_model(force_reload=False)

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            # Some word does not come with IDF. It will be excluded in lucky draw.
            word = tokens[token_idx]

            if word in self.model.w2idf:
                results.append(token_idx)

        return results

    def _get_aug_idxes(self, tokens):
        aug_cnt = self.generate_aug_cnt(len(tokens))
        word_idxes = self.pre_skip_aug(tokens)
        word_idxes = self.skip_aug(word_idxes, tokens)

        if len(word_idxes) == 0:
            if self.verbose > 0:
                exception = WarningException(name=WarningName.OUT_OF_VOCABULARY,
                                             code=WarningCode.WARNING_CODE_002, msg=WarningMessage.NO_WORD)
                exception.output()
            return None
        if len(word_idxes) < aug_cnt:
            aug_cnt = len(word_idxes)

        aug_probs = self.model.cal_tfidf(word_idxes, tokens)
        aug_idxes = []

        # It is possible that no token is picked. So re-try
        retry_cnt = 3
        possible_idxes = word_idxes.copy()
        for _ in range(retry_cnt):
            for i, p in zip(possible_idxes, aug_probs):
                if self.prob() < p:
                    aug_idxes.append(i)
                    possible_idxes.remove(i)

                    if len(possible_idxes) == aug_cnt:
                        break

        # If still cannot pick up, random pick index regrardless probability
        if len(aug_idxes) < aug_cnt:
            aug_idxes.extend(self.sample(possible_idxes, aug_cnt-len(aug_idxes)))

        aug_idxes = self.sample(aug_idxes, aug_cnt)

        return aug_idxes

    def insert(self, data):
        if not data or not data.strip():
            return data

        change_seq = 0
        doc = Doc(data, self.tokenizer(data))

        aug_idxes = self._get_random_aug_idxes(doc.get_original_tokens())
        if aug_idxes is None or len(aug_idxes) == 0:
            if self.include_detail:
                return data, []
            return data
        aug_idxes.sort(reverse=True)

        for aug_idx in aug_idxes:
            original_token = doc.get_token(aug_idx).orig_token.token
            candidate_tokens = self.model.predict(original_token, top_k=self.top_k)
            new_token = self.sample(candidate_tokens, 1)[0]

            if aug_idx == 0:
                new_token = new_token.capitalize()

            change_seq += 1
            doc.add_token(aug_idx, token=new_token, action=Action.INSERT,
                          change_seq=self.parent_change_seq + change_seq)

            if self.get_word_case(doc.get_token(0).get_latest_token().token) == 'capitalize':
                change_token = doc.get_token(1).get_latest_token().token.lower()
                doc.add_change_log(1, new_token=change_token, action=Action.INSERT,
                                   change_seq=self.parent_change_seq + change_seq)

        if self.include_detail:
            return self.reverse_tokenizer(doc.get_augmented_tokens()), doc.get_change_logs()
        else:
            return self.reverse_tokenizer(doc.get_augmented_tokens())

    def substitute(self, data):
        if not data or not data.strip():
            return data
            
        change_seq = 0
        doc = Doc(data, self.tokenizer(data))

        aug_idxes = self._get_aug_idxes(doc.get_original_tokens())
        if aug_idxes is None:
            return data

        for aug_idx in aug_idxes:
            original_token = doc.get_token(aug_idx).orig_token.token
            candidate_tokens = self.model.predict(original_token, top_k=self.top_k)
            substitute_token = self.sample(candidate_tokens, 1)[0]
            if aug_idx == 0:
                substitute_token = self.align_capitalization(original_token, substitute_token)

            change_seq += 1
            doc.add_change_log(aug_idx, new_token=substitute_token, action=Action.SUBSTITUTE,
                               change_seq=self.parent_change_seq + change_seq)

        if self.include_detail:
            return self.reverse_tokenizer(doc.get_augmented_tokens()), doc.get_change_logs()
        else:
            return self.reverse_tokenizer(doc.get_augmented_tokens())

    def get_model(self, force_reload=False):
        return init_tfidf_model(self.model_path, force_reload)
