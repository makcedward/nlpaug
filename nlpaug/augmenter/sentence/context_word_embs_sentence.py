"""
    Augmenter that apply operation (sentence level) to textual input based on contextual word embeddings.
"""

import os

from nlpaug.augmenter.sentence import SentenceAugmenter
import nlpaug.model.lang_models as nml
from nlpaug.util import Action, Doc
import nlpaug.util.text.tokenizer as text_tokenizer

CONTEXT_WORD_EMBS_SENTENCE_MODELS = {}


def init_context_word_embs_sentence_model(model_path, device, force_reload=False, temperature=1.0, top_k=None,
                                          top_p=None, optimize=None, silence=True):
    global CONTEXT_WORD_EMBS_SENTENCE_MODELS

    model_name = os.path.basename(model_path)
    if model_name in CONTEXT_WORD_EMBS_SENTENCE_MODELS and not force_reload:
        CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name].device = device
        if temperature != 1.0:
            CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name].temperature = temperature
        if top_k:
            CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name].top_k = top_k
        if top_p:
            CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name].top_p = top_p
        if optimize:
            CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name].optimize = optimize
        CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name].silence = silence
        return CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name]

    if 'xlnet' in model_path:
        model = nml.XlNet(model_path, device=device, temperature=temperature, top_k=top_k, top_p=top_p,
                          optimize=optimize, silence=True)
    elif 'gpt2' in model_path:
        model = nml.Gpt2(model_path, device=device, temperature=temperature, top_k=top_k, top_p=top_p,
                         optimize=optimize, silence=True)
    else:
        raise ValueError('Model name value is unexpected. Only support XLNet and GPT2 model.')

    CONTEXT_WORD_EMBS_SENTENCE_MODELS[model_name] = model
    return model


class ContextualWordEmbsForSentenceAug(SentenceAugmenter):
    # https://arxiv.org/pdf/1707.07328.pdf, https://arxiv.org/pdf/2003.02245.pdf
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
    :param str device: Default value is CPU. If value is CPU, it uses CPU for processing. If value is CUDA, it uses GPU
        for processing. Possible values include 'cuda' and 'cpu'. (May able to use other options)
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param obj optimize: Configuration for optimized process.
        `external_memory`: Persisting previous computed result for next prediction. Extra memory will be used in order
            to have shorter inference time. `gpt2` and `distilgpt2`are supported.
    :param bool silence: Default is True. transformers library will print out warning message when leveraing
        pre-trained model. Set True to disable the expected warning message.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.sentence as nas
    >>> aug = nas.ContextualWordEmbsForSentenceAug()
    """

    def __init__(self, model_path='distilgpt2', temperature=1.0, top_k=100, top_p=None,
                 name='ContextualWordEmbsForSentence_Aug',
                 device='cpu', force_reload=False, optimize=None, verbose=0, silence=True):
        super().__init__(
            action=Action.INSERT, name=name, tokenizer=None, stopwords=None, device=device,
            include_detail=False, parallelable=True, verbose=verbose)
        self.model_path = model_path
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.silence = silence

        self._init()
        self.model = self.get_model(
            model_path=model_path, device=device, force_reload=force_reload, temperature=temperature, top_k=top_k,
            top_p=top_p, optimize=optimize, silence=silence)
        self.device = self.model.device

    def _init(self):
        if 'xlnet' in self.model_path:
            self.model_type = 'xlnet'
        elif 'gpt2' in self.model_path:
            self.model_type = 'gpt2'
        else:
            self.model_type = ''

    def insert(self, data):
        if not data:
            return data

        if isinstance(data, list):
            all_data = data
        else:
            if data.strip() == '':
                return data

            all_data = [data]

        max_try = 30  # On average 30 should be enough to complete a sentence
        external_memories = [None] * len(all_data)
        augmented_texts = [''] * len(all_data)
        docs = [Doc()] * len(all_data)
        early_stops = [0] * len(all_data)
        change_seq = 0
        aug_idx = 0

        for _ in range(max_try):
            if sum(early_stops) == len(all_data):
                break

            aug_input_poses = [] # store which input augmented. No augmentation if genrated a sentence
            texts = []
            for i, d in enumerate(all_data):
                if early_stops[i] == 1:
                    continue

                aug_input_poses.append(i)
                augmented_text = augmented_texts[i]
                external_memory = external_memories[i]

                if external_memory is None:  # First step or does not enable optimization
                    text = d + augmented_text
                else:
                    text = ''

                # Mask token is needed for xlnet. No mask token for gpt2
                if self.model_type in ['xlnet']:
                    text += ' ' + self.model.MASK_TOKEN

                texts.append(text)

            outputs = self.model.predict(texts, n=1, external_memory=external_memory, 
                include_punctuation=True)

            for i, output in enumerate(outputs):
                aug_input_pos = aug_input_poses[i]

                # TODO:
                # if self.model.optimize['external_memory']:
                #     external_memory = outputs[1]

                # TODO: Alternative method better than dropout
                candidate = ''
                if len(output) == 1:
                    candidate = output[0]
                elif len(output) > 1:
                    candidate = self.sample(output, 1)[0]

                change_seq += 1
                docs[aug_input_pos].add_token(aug_idx, token='', action=Action.INSERT, change_seq=0)
                docs[aug_input_pos].update_change_log(aug_idx, token=self.model.clean(candidate), action=Action.INSERT,
                    change_seq=self.parent_change_seq + change_seq)
                aug_idx += 1

                # early stop if all input generated a sentence.
                if candidate in text_tokenizer.SENTENCE_SEPARATOR:
                    if self.model_type in ['gpt2']:
                        augmented_texts[aug_input_pos] += ' '
                    augmented_texts[aug_input_pos] += candidate
                    early_stops[aug_input_pos] = 1
                else:
                    if self.model_type in ['gpt2']:
                        augmented_texts[aug_input_pos] += ' '
                    augmented_texts[aug_input_pos] += candidate


        if self.model_type in ['gpt2']:
            results = [d + a for d, a in zip(all_data, augmented_texts)]
        elif self.model_type in ['xlnet']:
            results = [d + ' ' + self.model.tokenizer.convert_tokens_to_string(a) for d, a in zip(all_data, augmented_texts)]

        if isinstance(data, list):
            return results
        else:
            return results[0]

    @classmethod
    def get_model(cls, model_path, device='cuda', force_reload=False, temperature=1.0, top_k=None, top_p=0.0,
                  optimize=None, silence=True):
        return init_context_word_embs_sentence_model(model_path, device, force_reload, temperature, top_k, top_p,
                                                     optimize=optimize, silence=silence)
