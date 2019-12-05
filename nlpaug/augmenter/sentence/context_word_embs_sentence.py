"""
    Augmenter that apply operation (sentence level) to textual input based on contextual word embeddings.
"""

import os

from nlpaug.augmenter.sentence import SentenceAugmenter
import nlpaug.model.lang_models as nml
from nlpaug.util.action import Action

CONTEXT_WORD_EMBS_SENTENCE_MODELS = {}


def init_context_word_embs_sentence_model(model_path, device, force_reload=False, temperature=1.0, top_k=None,
                                          top_p=None, optimize=None):
    global CONTEXT_WORD_EMBS_SENTENCE_MODELS

    model_name = os.path.basename(model_path)
    if model_name in CONTEXT_WORD_EMBS_SENTENCE_MODELS and not force_reload:
        CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name].temperature = temperature
        CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name].top_k = top_k
        CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name].top_p = top_p
        return CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name]

    if 'xlnet' in model_path:
        model = nml.XlNet(model_path, device=device, temperature=temperature, top_k=top_k, top_p=top_p,
                          optimize=optimize)
    elif 'gpt2' in model_path:
        model = nml.Gpt2(model_path, device=device, temperature=temperature, top_k=top_k, top_p=top_p,
                         optimize=optimize)
    else:
        raise ValueError('Model name value is unexpected. Only support XLNet and GPT2 model.')

    CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name] = model
    return model


class ContextualWordEmbsForSentenceAug(SentenceAugmenter):
    # https://arxiv.org/pdf/1707.07328.pdf
    """
    Augmenter that leverage contextual word embeddings to find top n similar word for augmentation.

    :param str model_path: Model name or model path. It used transformers to load the model. Tested
        'xlnet-base-cased', 'gpt2', 'distilgpt2'. If you want to reduce inference time, you may select `distilgpt2`.
    :param float temperature: Controlling randomness. Default value is 1 and lower temperature results in less random
        behavior
    :param int top_k: Controlling lucky draw pool. Top k score token will be used for augmentation. Larger k, more
        token can be used. Default value is 100. If value is None which means using all possible tokens.
    :param float top_p: Controlling lucky draw pool. Top p of cumulative probability will be removed. Larger p, more
        token can be used. Default value is None which means using all possible tokens.
    :param str device: Use either cpu or gpu. Default value is None, it uses GPU if having. While possible values are
        'cuda' and 'cpu'.
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param obj optimize: Configuration for optimized process.
        `external_memory`: Persisting previous computed result for next prediction. Extra memory will be used in order
            to have shorter inference time. `gpt2` and `distilgpt2`are supported.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.sentence as nas
    >>> aug = nas.ContextualWordEmbsForSentenceAug()
    """

    def __init__(self, model_path='distilgpt2', temperature=1.0, top_k=100, top_p=None,
                 name='ContextualWordEmbsForSentence_Aug',
                 device=None, force_reload=False, optimize=None, verbose=0):
        super().__init__(
            action=Action.INSERT, name=name, tokenizer=None, stopwords=None, device=device, verbose=verbose)
        self.model_path = model_path
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self._init()
        self.model = self.get_model(
            model_path=model_path, device=device, force_reload=force_reload, temperature=temperature, top_k=top_k,
            top_p=top_p, optimize=optimize)
        self.device = self.model.device

    def _init(self):
        if 'xlnet' in self.model_path:
            self.model_type = 'xlnet'
        elif 'gpt2' in self.model_path:
            self.model_type = 'gpt2'
        else:
            self.model_type = ''

    def insert(self, data):
        if data is None or data == '' or data.strip() == '':
            return data

        max_try = 30  # On average 30 should be enough to complete a sentence
        external_memory = None
        augmented_text = ''
        new_word = ''

        for _ in range(max_try):
            if external_memory is None:  # First step or does not enable optimization
                text = data + augmented_text
            else:
                text = new_word

            # Mask token is needed for xlnet. No mask token for gpt2
            if self.model_type in ['xlnet']:
                text += ' ' + self.model.MASK_TOKEN

            outputs = self.model.predict(text, n=1, external_memory=external_memory)
            results = outputs[0]
            if results is None:
                continue

            if self.model.optimize['external_memory']:
                external_memory = outputs[1]

            new_word, proba = results[0]
            if new_word in self.SENTENCE_SEPARATOR:
                augmented_text += new_word
                break

            augmented_text += ' ' + new_word

        return data + ' ' + self.model.clean(augmented_text)

    @classmethod
    def get_model(cls, model_path, device='cuda', force_reload=False, temperature=1.0, top_k=None, top_p=0.0,
                  optimize=None):
        return init_context_word_embs_sentence_model(model_path, device, force_reload, temperature, top_k, top_p,
                                                     optimize=optimize)
