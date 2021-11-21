"""
    Augmenter that apply operation to textual input based on word embeddings.
"""

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action, Doc
import nlpaug.model.word_embs as nmw


WORD_EMBS_MODELS = {}
model_types = ['word2vec', 'glove', 'fasttext']


def init_word_embs_model(model_path, model_type, force_reload=False, top_k=None, skip_check=False):
    global WORD_EMBS_MODELS

    if model_type in WORD_EMBS_MODELS and not force_reload:
        WORD_EMBS_MODELS[model_type].top_k = top_k
        return WORD_EMBS_MODELS[model_type]

    if model_type == 'word2vec':
        model = nmw.Word2vec(top_k=top_k, skip_check=skip_check)
        model.read(model_path)
    elif model_type == 'glove':
        model = nmw.GloVe(top_k=top_k, skip_check=skip_check)
        model.read(model_path)
    elif model_type == 'fasttext':
        model = nmw.Fasttext(top_k=top_k, skip_check=skip_check)
        model.read(model_path)
    else:
        raise ValueError('Model type value is unexpected. Expected values include {}'.format(model_types))

    WORD_EMBS_MODELS[model_type] = model
    return model


class WordEmbsAug(WordAugmenter):
    # https://aclweb.org/anthology/D15-1306, https://arxiv.org/pdf/1804.07998.pdf, https://arxiv.org/pdf/1509.01626.pdf
    # https://arxiv.org/ftp/arxiv/papers/1812/1812.04718.pdf
    """
    Augmenter that leverage word embeddings to find top n similar word for augmentation.

    :param str model_type: Model type of word embeddings. Expected values include 'word2vec', 'glove' and 'fasttext'.
    :param str model_path: Downloaded model directory. Either model_path or model is must be provided
    :param obj model: Pre-loaded model (e.g. model class is nlpaug.model.word_embs.nmw.Word2vec(), nlpaug.model.word_embs.nmw.Glove()
        or nlpaug.model.word_embs.nmw.Fasttext())
    :param str action: Either 'insert or 'substitute'. If value is 'insert', a new word will be injected to random
        position according to word embeddings calculation. If value is 'substitute', word will be replaced according
        to word embeddings calculation
    :param int top_k: Controlling lucky draw pool. Top k score token will be used for augmentation. Larger k, more
        token can be used. Default value is 100. If value is None which means using all possible tokens. This attribute will
        be ignored when using "insert" action.
    :param float aug_p: Percentage of word will be augmented.
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result
        from aug_p. Otherwise, using aug_max.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param bool force_reload: If True, model will be loaded every time while it takes longer time for initialization.
    :param bool skip_check: Default is False. If True, no validation for size of vocabulary embedding.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.WordEmbsAug(model_type='word2vec', model_path='.')
    """

    def __init__(self, model_type, model_path='.', model=None, action=Action.SUBSTITUTE,
        name='WordEmbs_Aug', aug_min=1, aug_max=10, aug_p=0.3, top_k=100, n_gram_separator='_',
        stopwords=None, tokenizer=None, reverse_tokenizer=None, force_reload=False, stopwords_regex=None,
        verbose=0, skip_check=False):
        super().__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, device='cpu', verbose=verbose,
            stopwords_regex=stopwords_regex, include_detail=False)

        self.model_type = model_type
        self.model_path = model_path

        self.top_k = top_k
        self.n_gram_separator = n_gram_separator

        self.pre_validate()

        if model is None:
            self.model = self.get_model(model_path=model_path, model_type=model_type, force_reload=force_reload,
                                        top_k=self.top_k, skip_check=skip_check)
        else:
            self.model = model

    def pre_validate(self):
        if self.model_type not in model_types:
            raise ValueError('Model type value is unexpected. Expected values include {}'.format(model_types))

    @classmethod
    def get_model(cls, model_path, model_type, force_reload=False, top_k=100, skip_check=False):
        return init_word_embs_model(model_path, model_type, force_reload, top_k=top_k, skip_check=skip_check)

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            # Some words do not come with vector. It will be excluded in lucky draw.
            word = tokens[token_idx]

            if word in self.model.get_vocab():
                results.append(token_idx)

        return results

    def insert(self, data):
        if not data or not data.strip():
            return data

        change_seq = 0
        doc = Doc(data, self.tokenizer(data))

        aug_idxes = self._get_random_aug_idxes(doc.get_original_tokens())
        if not aug_idxes:
            if self.include_detail:
                return data, []
            return data
        aug_idxes.sort(reverse=True)

        for aug_idx in aug_idxes:
            new_token = self.sample(self.model.get_vocab(), 1)[0]
            if self.n_gram_separator in new_token:
                new_token = new_token.split(self.n_gram_separator)[0]

            change_seq += 1
            doc.add_token(aug_idx, token=new_token, action=Action.INSERT,
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
        if aug_idxes is None or len(aug_idxes) == 0:
            if self.include_detail:
                return data, []
            return data

        for aug_idx in aug_idxes:
            original_token = doc.get_token(aug_idx).get_latest_token().token
            candidate_tokens = self.model.predict(original_token, n=1)
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
