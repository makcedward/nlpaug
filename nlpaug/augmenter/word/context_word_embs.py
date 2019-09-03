"""
    Augmenter that apply operation (word level) to textual input based on contextual word embeddings.
"""

from nlpaug.augmenter.word import WordAugmenter
import nlpaug.model.lang_models as nml

BERT_MODEL = {}
XLNET_MODEL = {}


def init_bert_model(model_path, device, force_reload=False):
    # Load model once at runtime

    global BERT_MODEL
    if BERT_MODEL and not force_reload:
        return BERT_MODEL

    bert_model = nml.Bert(model_path, device=device)
    bert_model.model.eval()
    BERT_MODEL = bert_model

    return bert_model


def init_xlnet_model(model_path, device, force_reload=False):
    # Load model once at runtime

    global XLNET_MODEL
    if XLNET_MODEL and not force_reload:
        return XLNET_MODEL

    xlnet_model = nml.XlNet(model_path, device=device)
    xlnet_model.model.eval()
    XLNET_MODEL = xlnet_model

    return xlnet_model


class ContextualWordEmbsAug(WordAugmenter):
    """
    Augmenter that leverage contextual word embeddings to find top n similar word for augmentation.

    :param str model_path: Model name or model path. It used pytorch-transformer to load the model. Tested
        'bert-base-uncased', 'xlnet-base-cased'.
    :param str action: Either 'insert or 'substitute'. If value is 'insert', a new word will be injected to random
        position according to contextual word embeddings calculation. If value is 'substitute', word will be replaced
        according to contextual embeddings calculation
    :param int aug_min: Minimum number of word will be augmented.
    :param float aug_p: Percentage of word will be augmented.
    :param int aug_n: Top n similar word for lucky draw
    :param list stopwords: List of words which will be skipped from augment operation.
    :param str device: Use either cpu or gpu. Default value is 'cuda' while possible values are 'cuda' and 'cpu'.
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.ContextualWordEmbsAug()
    """

    def __init__(self, model_path='bert-base-uncased', action="substitute", name='ContextualWordEmbs_Aug',
                 aug_min=1, aug_p=0.3, aug_n=5, stopwords=None, device='cuda', force_reload=False, verbose=0):
        super().__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, tokenizer=None, stopwords=stopwords,
            verbose=verbose)
        self.model_path = model_path
        self.aug_n = aug_n
        self.device = device
        self.model = self.get_model(model_path=model_path, device=device, force_reload=force_reload)
        self.tokenizer = self.model.tokenizer.tokenize

    def insert(self, data):
        # Pick target word for augmentation
        tokens = data.split(' ')
        aug_idxes = self._get_random_aug_idxes(tokens)
        aug_idxes.sort(reverse=True)

        for aug_idx in aug_idxes:
            tokens.insert(aug_idx, self.model.MASK_TOKEN)
            masked_text = ' '.join(tokens)

            candidates = self.model.predict(masked_text, target_word=None, top_n=self.aug_n)
            new_word, prob = self.sample(candidates, 1)[0]
            tokens[aug_idx] = new_word

        return ' '.join(tokens)

    def substitute(self, data):
        # Pick target word for augmentation
        tokens = data.split(' ')
        aug_idxes = self._get_aug_idxes(tokens)

        for aug_idx in aug_idxes:
            original_word = tokens[aug_idx]
            tokens[aug_idx] = self.model.MASK_TOKEN
            masked_text = ' '.join(tokens)

            candidates = self.model.predict(masked_text, target_word=original_word, top_n=self.aug_n)
            substitute_word, prob = self.sample(candidates, 1)[0]

            tokens[aug_idx] = substitute_word

        results = []
        for src, dest in zip(data.split(' '), tokens):
            results.append(self.align_capitalization(src, dest))

        return ' '.join(results)

    @classmethod
    def get_model(self, model_path, device='cuda', force_reload=False):
        if 'bert' in model_path:
            return init_bert_model(model_path, device, force_reload)
        if 'xlnet' in model_path:
            return init_xlnet_model(model_path, device, force_reload)

        raise ValueError('Model name value is unexpected.. Only support bert and xlnet model.')
