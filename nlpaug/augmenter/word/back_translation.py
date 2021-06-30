"""
    Augmenter that apply operation (word level) to textual input based on back translation.
"""

import string
import os
import torch

from nlpaug.augmenter.word import WordAugmenter
import nlpaug.model.lang_models as nml

BACK_TRANSLATION_MODELS = {}


def init_back_translatoin_model(model_src, from_model_name, to_model_name, device, force_reload=False,
                                batch_size=32, max_tokens=None):
    global BACK_TRANSLATION_MODELS

    model_name = '_'.join([model_src, from_model_name, to_model_name])
    if model_name in BACK_TRANSLATION_MODELS and not force_reload:
        BACK_TRANSLATION_MODELS[model_name].device = device

        return BACK_TRANSLATION_MODELS[model_name]
    if model_src == 'huggingface':
        model = nml.MtTransformers(src_model_name=from_model_name, tgt_model_name=to_model_name, device=device,
                                   batch_size=batch_size, max_tokens=max_tokens)
    # elif model_src == 'fairseq':
    #     model = nml.Fairseq(from_model_name=from_model_name, from_model_checkpt=from_model_checkpt, 
    #         to_model_name=to_model_name, to_model_checkpt=to_model_checkpt, 
    #         tokenzier_name=tokenzier_name, bpe_name=bpe_name, is_load_from_github=is_load_from_github, 
    #         device=device)

    BACK_TRANSLATION_MODELS[model_name] = model
    return model


class BackTranslationAug(WordAugmenter):
    # https://arxiv.org/pdf/1511.06709.pdf
    """
    Augmenter that leverage two translation models for augmentation. For example, the source is English. This
    augmenter translate source to German and translating it back to English. For detail, you may visit
    https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28

    :param str from_model_name: Any model from https://huggingface.co/models?filter=translation&search=Helsinki-NLP. As
        long as from_model_name is pair with to_model_name. For example, from_model_name is English to Japanese,
        then to_model_name should be Japanese to English.
    :param str to_model_name: Any model from https://huggingface.co/models?filter=translation&search=Helsinki-NLP.
    :param str device: Default value is CPU. If value is CPU, it uses CPU for processing. If value is CUDA, it uses GPU
        for processing. Possible values include 'cuda' and 'cpu'. (May able to use other options)
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.BackTranslationAug()
    """

    def __init__(self, from_model_name='Helsinki-NLP/opus-mt-en-de', to_model_name='Helsinki-NLP/opus-mt-de-en',
        name='BackTranslationAug', device='cpu', force_reload=False, verbose=0, batch_size=32, max_tokens=None):
        super().__init__(
            action='substitute', name=name, aug_p=None, aug_min=None, aug_max=None, tokenizer=None,
            device=device, verbose=verbose, include_detail=False, parallelable=True)

        # migrate from fairseq to huggingface library
        self.model_src = 'huggingface'

        self.model = self.get_model(model_src=self.model_src,
            from_model_name=from_model_name, to_model_name=to_model_name, device=device,
            batch_size=batch_size, max_tokens=max_tokens
        )
        self.device = self.model.device

    def substitute(self, data):
        if not data:
            return data

        augmented_text = self.model.predict(data)
        return augmented_text

    @classmethod
    def get_model(cls, model_src, from_model_name, to_model_name, device='cuda', force_reload=False,
                  batch_size=32, max_tokens=None):
        return init_back_translatoin_model(model_src, from_model_name, to_model_name, device,
            force_reload, batch_size, max_tokens)

    @classmethod
    def clear_cache(cls):
        global BACK_TRANSLATION_MODELS
        BACK_TRANSLATION_MODELS = {}
