"""
    Augmenter that apply operation (sentence level) to textual input based on contextual word embeddings.
"""

import os

from nlpaug.augmenter.sentence import SentenceAugmenter
import nlpaug.model.lang_models as nml
from nlpaug.util import Action, Doc
import nlpaug.util.text.tokenizer as text_tokenizer

CONTEXT_WORD_EMBS_SENTENCE_MODELS = {}


def init_context_word_embs_sentence_model(model_path, device, force_reload=False, 
    min_length=100, max_length=300, batch_size=32, temperature=1.0, top_k=50, top_p=0.9, 
    silence=True):

    global CONTEXT_WORD_EMBS_SENTENCE_MODELS

    model_name = '_'.join([os.path.basename(model_path), str(device)])
    if model_name in CONTEXT_WORD_EMBS_SENTENCE_MODELS and not force_reload:
        CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name].min_length = min_length
        CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name].max_length = max_length
        CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name].temperature = temperature
        CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name].top_k = top_k
        CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name].top_p = top_p
        CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name].batch_size = batch_size
        CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name].silence = silence
        return CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name]

    model = nml.TextGenTransformers(model_path, device=device, min_length=min_length, max_length=max_length, 
        temperature=temperature, top_k=top_k, top_p=top_p, batch_size=batch_size)

    CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name] = model
    return model


class ContextualWordEmbsForSentenceAug(SentenceAugmenter):
    # https://arxiv.org/pdf/1707.07328.pdf, https://arxiv.org/pdf/2003.02245.pdf
    """
    Augmenter that leverage contextual word embeddings to find top n similar word for augmentation.

    :param str model_path: Model name or model path. It used transformers to load the model. Tested
        'gpt2', 'distilgpt2'. 
    :param int batch_size: Batch size.
    :param int min_length: The min length of output text.
    :param int max_length: The max length of output text.
    :param float temperature: The value used to module the next token probabilities.
    :param int top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
    :param float top_p: If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or
        higher are kept for generation.
    :param str device: Default value is CPU. If value is CPU, it uses CPU for processing. If value is CUDA, it uses GPU
        for processing. Possible values include 'cuda' and 'cpu'. (May able to use other options)
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param bool silence: Default is True. transformers library will print out warning message when leveraing
        pre-trained model. Set True to disable the expected warning message.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.sentence as nas
    >>> aug = nas.ContextualWordEmbsForSentenceAug()
    """

    def __init__(self, model_path='gpt2', name='ContextualWordEmbsForSentence_Aug',
        min_length=100, max_length=500, batch_size=32, temperature=1.0, top_k=50, top_p=0.9, 
        device='cpu', force_reload=False, silence=True):
        super().__init__(
            action=Action.INSERT, name=name, tokenizer=None, stopwords=None, device=device,
            include_detail=False)
        self.model_path = model_path

        self.model = self.get_model(
            model_path=model_path, device=device, force_reload=force_reload, batch_size=batch_size,
            min_length=min_length, max_length=max_length, temperature=temperature,
            top_k=top_k, top_p=top_p, silence=silence)
        self.device = self.model.device

    def check_model_type(self):
        if 'xlnet' in self.model_path.lower():
            return 'xlnet'
        elif 'gpt2' in self.model_path.lower():
            return 'gpt2'
        return ''

    def insert(self, data):
        if not data:
            return data

        if isinstance(data, list):
            all_data = data
        else:
            if data.strip() == '':
                return data

            all_data = [data]

        return self.model.predict(all_data)

    @classmethod
    def get_model(cls, model_path, device='cuda', force_reload=False, min_length=100, 
        max_length=300, batch_size=32, temperature=1.0, top_k=50, top_p=0.9, silence=True):
        return init_context_word_embs_sentence_model(model_path, device, force_reload, 
            batch_size=batch_size, min_length=min_length, max_length=max_length, 
            temperature=temperature, top_k=top_k, top_p=top_p, silence=silence)
