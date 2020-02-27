"""
    Augmenter that apply operation (word level) to textual input based on contextual word embeddings.
"""

import string
import os

from nlpaug.augmenter.word import WordAugmenter
import nlpaug.model.lang_models as nml

CONTEXT_WORD_EMBS_MODELS = {}


def init_context_word_embs_model(model_path, device, force_reload=False, temperature=1.0, top_k=None, top_p=None,
                                 optimize=None):
    global CONTEXT_WORD_EMBS_MODELS

    model_name = os.path.basename(model_path)
    if model_name in CONTEXT_WORD_EMBS_MODELS and not force_reload:
        CONTEXT_WORD_EMBS_MODELS[model_name].temperature = temperature
        CONTEXT_WORD_EMBS_MODELS[model_name].top_k = top_k
        CONTEXT_WORD_EMBS_MODELS[model_name].top_p = top_p
        return CONTEXT_WORD_EMBS_MODELS[model_name]

    if 'distilbert' in model_path:
        model = nml.DistilBert(model_path, device=device, temperature=temperature, top_k=top_k, top_p=top_p)
    elif 'roberta' in model_path:
        model = nml.Roberta(model_path, device=device, temperature=temperature, top_k=top_k, top_p=top_p)
    elif 'bert' in model_path:
        model = nml.Bert(model_path, device=device, temperature=temperature, top_k=top_k, top_p=top_p)
    elif 'xlnet' in model_path:
        model = nml.XlNet(model_path, device=device, temperature=temperature, top_k=top_k, top_p=top_p, optimize=optimize)
    else:
        raise ValueError('Model name value is unexpected. Only support BERT, DistilBERT, RoBERTa and XLNet model.')

    CONTEXT_WORD_EMBS_MODELS[model_name] = model
    return model


class ContextualWordEmbsAug(WordAugmenter):
    # https://arxiv.org/pdf/1805.06201.pdf
    """
    Augmenter that leverage contextual word embeddings to find top n similar word for augmentation.

    :param str model_path: Model name or model path. It used transformers to load the model. Tested
        'bert-base-uncased', 'bert-base-cased', 'distilbert-base-uncased', 'roberta-base', 'distilroberta-base',
        'xlnet-base-cased'.
    :param str action: Either 'insert or 'substitute'. If value is 'insert', a new word will be injected to random
        position according to contextual word embeddings calculation. If value is 'substitute', word will be replaced
        according to contextual embeddings calculation
    :param float temperature: Controlling randomness. Default value is 1 and lower temperature results in less random
        behavior
    :param int top_k: Controlling lucky draw pool. Top k score token will be used for augmentation. Larger k, more
        token can be used. Default value is 100. If value is None which means using all possible tokens.
    :param float top_p: Controlling lucky draw pool. Top p of cumulative probability will be removed. Larger p, more
        token can be used. Default value is None which means using all possible tokens.
    :param float aug_p: Percentage of word will be augmented.
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result from
        aug_p. Otherwise, using aug_max.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param bool skip_unknown_word: Do not substitute unknown word (e.g. AAAAAAAAAAA)
    :param str device: Use either cpu or gpu. Default value is None, it uses GPU if having. While possible values are
        'cuda' and 'cpu'.
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param bool optimize: If true, optimized process will be executed. For example, GPT2 will use "return_past" to
        reduce inference time.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.ContextualWordEmbsAug()
    """

    def __init__(self, model_path='bert-base-uncased', action="substitute", temperature=1.0, top_k=100, top_p=None,
                 name='ContextualWordEmbs_Aug', aug_min=1, aug_max=10, aug_p=0.3, stopwords=None,
                 skip_unknown_word=False, device=None, force_reload=False, optimize=None, stopwords_regex=None,
                 verbose=0):
        super().__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, tokenizer=None,
            device=device, stopwords=stopwords, verbose=verbose, stopwords_regex=stopwords_regex)
        self.model_path = model_path
        self.skip_unknown_word = skip_unknown_word
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self._init()
        self.model = self.get_model(
            model_path=model_path, device=device, force_reload=force_reload, temperature=temperature, top_k=top_k,
            top_p=top_p, optimize=optimize)
        # Override stopwords
        if stopwords is not None and self.model_type in ['xlnet', 'roberta']:
            stopwords = [self.stopwords]
        self.device = self.model.device

        """
            TODO: Reserve 2 spaces (e.g. [CLS], [SEP]) is not enough as it hit CUDA error in batch processing mode.
            Therefore, forcing to reserve 5 times of reserved spaces (i.e. 5)
        """
        self.max_num_token = self.model.model.config.max_position_embeddings - 2 * 5

    def _init(self):
        if 'xlnet' in self.model_path:
            self.model_type = 'xlnet'
        elif 'distilbert' in self.model_path:
            self.model_type = 'distilbert'
        elif 'roberta' in self.model_path:
            self.model_type = 'roberta'
        elif 'bert' in self.model_path:
            self.model_type = 'bert'
        else:
            self.model_type = ''

    def skip_aug(self, token_idxes, tokens):
        if not self.skip_unknown_word:
            return super().skip_aug(token_idxes, tokens)

        found_suffix = False

        for token_idx in reversed(token_idxes[:]):
            if self.model_type in ['bert', 'distilbert'] and self.model.SUBWORD_PREFIX in tokens[token_idx]:
                token_idxes.remove(token_idx)
                found_suffix = True
                continue
            if self.model_type in ['xlnet', 'roberta'] and self.model.SUBWORD_PREFIX not in tokens[token_idx] \
                    and tokens[token_idx] not in string.punctuation:
                token_idxes.remove(token_idx)
                found_suffix = True
                continue

            # Do not augment unknown word. For example abcde will split into "abc" and "##de" in BERT. Will ignore it
            if found_suffix:
                token_idxes.remove(token_idx)
                found_suffix = False

        return token_idxes

    def split_text(self, data):
        tokens = self.model.tokenizer.tokenize(data)

        if self.model.model.config.max_position_embeddings == -1:  # e.g. No max length restriction for XLNet
            return data, None, tokens, None  # Head text, tail text, head token, tail token

        head_text = self.model.tokenizer.convert_tokens_to_string(tokens[:self.max_num_token]).strip()
        tail_text = None
        if len(tokens) >= self.max_num_token:
            tail_text = self.model.tokenizer.convert_tokens_to_string(tokens[self.max_num_token:]).strip()

        return head_text, tail_text, tokens[:self.max_num_token], tokens[self.max_num_token:]

    def insert(self, data):
        # If length of input is larger than max allowed input, only augment heading part
        head_text, tail_text, head_tokens, tail_tokens = self.split_text(data)
        # Pick target word for augmentation
        aug_idxes = self._get_aug_idxes(head_tokens)
        if aug_idxes is None or len(aug_idxes) == 0:
            return data
        aug_idxes.sort(reverse=True)

        for aug_idx in aug_idxes:
            if self.model_type in ['xlnet', 'roberta']:
                head_tokens.insert(aug_idx, self.model.SUBWORD_PREFIX + self.model.MASK_TOKEN)  # Adding prefix for space
            else:
                head_tokens.insert(aug_idx, self.model.MASK_TOKEN)

            masked_text = self.model.tokenizer.convert_tokens_to_string(head_tokens).strip()

            # https://github.com/makcedward/nlpaug/issues/68
            retry_cnt = 3
            new_word, prob = None, None
            for _ in range(retry_cnt):
                outputs = self.model.predict(masked_text, target_word=None, n=1)
                candidates = outputs[0]
                if candidates is None:
                    continue

                if len(candidates) > 0:
                    new_word, prob = self.sample(candidates, 1)[0]
                    break

            # TODO: Alternative method better than dropout
            if new_word is None:
                new_word = ''

            head_tokens[aug_idx] = new_word

            # Early stop if number of token exceed max number
            if len(head_tokens) > self.max_num_token:
                break

        augmented_text = self.model.tokenizer.convert_tokens_to_string(head_tokens)
        if tail_text is not None:
            augmented_text += ' ' + tail_text

        return augmented_text

    def substitute(self, data):
        # If length of input is larger than max allowed input, only augment heading part
        head_text, tail_text, head_tokens, tail_tokens = self.split_text(data)
        # Pick target word for augmentation

        if self.model_type in ['xlnet', 'roberta']:
            # xlent and roberta tokens include prefix (e.g. ▁ or Ġ')
            cleaned_head_tokens = [t.replace(self.model.SUBWORD_PREFIX, '') for t in head_tokens]
        else:
            cleaned_head_tokens = head_tokens
        aug_idxes = self._get_aug_idxes(cleaned_head_tokens)
        if aug_idxes is None or len(aug_idxes) == 0:
            return data
        aug_idxes.sort(reverse=True)

        for i, aug_idx in enumerate(aug_idxes):
            original_word = head_tokens[aug_idx]
            if self.model_type in ['xlnet', 'roberta']:
                head_tokens[aug_idx] = self.model.SUBWORD_PREFIX + self.model.MASK_TOKEN  # Adding prefix for space
            else:
                head_tokens[aug_idx] = self.model.MASK_TOKEN

            # remove continuous subword
            to_remove_idxes = []
            for j in range(aug_idx+1, len(head_tokens)):
                if self.model_type in ['bert', 'distilbert'] and self.model.SUBWORD_PREFIX in head_tokens[j]:
                    to_remove_idxes.append(j)
                elif self.model_type in ['xlnet', 'roberta'] and self.model.SUBWORD_PREFIX not in head_tokens[j]:
                    to_remove_idxes.append(j)
                else:
                    break
            [head_tokens.pop(j) for j in reversed(to_remove_idxes)]

            masked_text = self.model.tokenizer.convert_tokens_to_string(head_tokens).strip()

            substitute_word, prob = None, None
            # https://github.com/makcedward/nlpaug/pull/51
            retry_cnt = 3
            for _ in range(retry_cnt):
                outputs = self.model.predict(masked_text, target_word=original_word, n=1+_)
                candidates = outputs[0]

                if candidates is None:
                    continue

                # Filter out unused candidates (transfomers may return [unused123])
                candidates = [c for c in candidates if '[unused' not in c[0] and ']' not in c[0]]

                if len(candidates) > 0:
                    substitute_word, prob = self.sample(candidates, 1)[0]
                    break

            # TODO: Alternative method better than dropout
            if substitute_word is None:
                substitute_word = ''

            if self.model_type in ['xlnet', 'roberta']:
                head_tokens[aug_idx] = self.model.SUBWORD_PREFIX + substitute_word  # Adding prefix for space
            else:
                head_tokens[aug_idx] = substitute_word

            # Early stop if number of token exceed max number
            if len(head_tokens) > self.max_num_token:
                break

        augmented_text = self.model.tokenizer.convert_tokens_to_string(head_tokens)
        if tail_text is not None:
            augmented_text += ' ' + tail_text

        return augmented_text

    @classmethod
    def get_model(cls, model_path, device='cuda', force_reload=False, temperature=1.0, top_k=None, top_p=0.0,
                  optimize=None):
        return init_context_word_embs_model(model_path, device, force_reload, temperature, top_k, top_p, optimize)
