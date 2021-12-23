"""
    Augmenter that apply operation (sentence level) to textual input based on contextual word embeddings.
"""

import os

from nlpaug.augmenter.sentence import SentenceAugmenter
import nlpaug.model.lang_models as nml
from nlpaug.util import Action, Doc
import nlpaug.util.text.tokenizer as text_tokenizer

CONTEXT_WORD_EMBS_SENTENCE_MODELS = {}


def init_context_word_embs_sentence_model(model_path, model_type, device, force_reload=False, 
    min_length=100, max_length=300, batch_size=32, temperature=1.0, top_k=50, top_p=0.9, 
    silence=True, use_custom_api=True):

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

    if use_custom_api:
        if model_type == 'xlnet':
            model = nml.XlNet(model_path, device=device, temperature=temperature, top_k=top_k, 
                              optimize=None, silence=True)
        elif model_type == 'gpt2':
            model = nml.Gpt2(model_path, device=device, temperature=temperature, top_k=top_k, 
                             optimize=None, silence=True)
        else:
            raise ValueError('Model name value is unexpected. Only support XLNet and GPT2 model.')
    else:
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
    :param str model_type: Type of model. For XLNet model, use 'xlnet'. For GPT2 or distilgpt2 model, use 'gpt'. If 
        no value is provided, will determine from model name.
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

    def __init__(self, model_path='gpt2', model_type='', name='ContextualWordEmbsForSentence_Aug',
        min_length=100, max_length=500, batch_size=32, temperature=1.0, top_k=50, top_p=0.9, 
        device='cpu', force_reload=False, silence=True, use_custom_api=True):
        super().__init__(
            action=Action.INSERT, name=name, tokenizer=None, stopwords=None, device=device,
            include_detail=False)
        self.model_path = model_path
        self.model_type = model_type if model_type != '' else self.check_model_type() 
        self.use_custom_api = use_custom_api

        self.model = self.get_model(
            model_path=model_path, model_type=self.model_type,
            device=device, force_reload=force_reload, batch_size=batch_size,
            min_length=min_length, max_length=max_length, temperature=temperature,
            top_k=top_k, top_p=top_p, silence=silence, use_custom_api=use_custom_api)
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

        if self.use_custom_api:
            return self._custom_insert(all_data)
        else:
            return self._native_insert(all_data)

    def _custom_insert(self, all_data):
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

        return results

    def _native_insert(self, all_data):
        return self.model.predict(all_data)

    @classmethod
    def get_model(cls, model_path, model_type, device='cuda', force_reload=False, min_length=100, 
        max_length=300, batch_size=32, temperature=1.0, top_k=50, top_p=0.9, silence=True, 
        use_custom_api=True):
        return init_context_word_embs_sentence_model(model_path, model_type, device, force_reload, 
            batch_size=batch_size, min_length=min_length, max_length=max_length, 
            temperature=temperature, top_k=top_k, top_p=top_p, silence=silence, 
            use_custom_api=use_custom_api)
