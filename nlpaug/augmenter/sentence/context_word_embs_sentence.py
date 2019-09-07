"""
    Augmenter that apply operation (sentence level) to textual input based on contextual word embeddings.
"""

from nlpaug.augmenter.sentence import SentenceAugmenter
import nlpaug.model.lang_models as nml
from nlpaug.util.action import Action

XLNET_MODEL = {}
GPT2_MODEL = {}


def init_xlnet_model(model_path, device, force_reload=False):
    # Load model once at runtime
    global XLNET_MODEL
    if XLNET_MODEL and not force_reload:
        return XLNET_MODEL

    xlnet_model = nml.XlNet(model_path, device=device)
    xlnet_model.model.eval()
    XLNET_MODEL = xlnet_model

    return xlnet_model


def init_gpt2_model(model_path, device, force_reload=False):
    # Load model once at runtime
    global GPT2_MODEL
    if GPT2_MODEL and not force_reload:
        return GPT2_MODEL

    gpt2_model = nml.Gpt2(model_path, device=device)
    gpt2_model.model.eval()
    GPT2_MODEL = gpt2_model

    return gpt2_model


class ContextualWordEmbsForSentenceAug(SentenceAugmenter):
    """
    Augmenter that leverage contextual word embeddings to find top n similar word for augmentation.

    :param str model_path: Model name or model path. It used pytorch-transformer to load the model. Tested
        'xlnet-base-cased', 'gpt2'.
    :param str device: Use either cpu or gpu. Default value is 'cuda' while possible values are 'cuda' and 'cpu'.
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.ContextualWordEmbsForSentenceAug()
    """

    def __init__(self, model_path='xlnet-base-cased', name='ContextualWordEmbsForSentence_Aug',
                 device='cuda', force_reload=False, verbose=0):
        super().__init__(
            action=Action.INSERT, name=name, aug_p=0.3, aug_min=1, tokenizer=None, stopwords=None,
            verbose=verbose)
        self.model_path = model_path
        self.device = device

        self._init()
        self.model = self.get_model(model_path=model_path, device=device, force_reload=force_reload)
        self.tokenizer = self.model.tokenizer.tokenize

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

        max_try = 100
        augmented_text = ''

        for _ in range(max_try):
            text = data + augmented_text
            # Mask token is needed for xlnet. No mask token for gpt2
            if self.model_type in ['xlnet']:
                text += ' ' + self.model.MASK_TOKEN

            results = self.model.predict(text, top_n=1)
            # if len(results) > 0:
            new_word, proba = results[0]

            if new_word in self.SENTENCE_SEPARATOR:
                augmented_text += new_word
                break

            augmented_text += ' ' + new_word

        return data + ' ' + self.model.clean(augmented_text)

    @classmethod
    def get_model(self, model_path, device='cuda', force_reload=False):
        if 'xlnet' in model_path:
            return init_xlnet_model(model_path, device, force_reload)
        if 'gpt2' in model_path:
            return init_gpt2_model(model_path, device, force_reload)

        raise ValueError('Model name value is unexpected. Only support xlnet and gpt2 model.')
