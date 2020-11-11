"""
    Augmenter that apply operation (word level) to textual input based on contextual word embeddings.
"""

import string
import os

from nlpaug.augmenter.word import WordAugmenter
import nlpaug.model.lang_models as nml
from nlpaug.util import Action, Doc

CONTEXT_WORD_EMBS_MODELS = {}


def init_context_word_embs_model(model_path, device, force_reload=False, temperature=1.0, top_k=None, top_p=None,
                                 optimize=None, silence=True):
    global CONTEXT_WORD_EMBS_MODELS

    model_name = os.path.basename(model_path)
    if model_name in CONTEXT_WORD_EMBS_MODELS and not force_reload:
        CONTEXT_WORD_EMBS_MODELS[model_name].device = device
        if temperature != 1.0:
            CONTEXT_WORD_EMBS_MODELS[model_name].temperature = temperature
        if top_k:
            CONTEXT_WORD_EMBS_MODELS[model_name].top_k = top_k
        if top_p:
            CONTEXT_WORD_EMBS_MODELS[model_name].top_p = top_p
        CONTEXT_WORD_EMBS_MODELS[model_name].silence = silence
        return CONTEXT_WORD_EMBS_MODELS[model_name]

    if 'distilbert' in model_path.lower():
        model = nml.DistilBert(model_path, device=device, temperature=temperature, top_k=top_k, top_p=top_p, silence=silence)
    elif 'roberta' in model_path.lower():
        model = nml.Roberta(model_path, device=device, temperature=temperature, top_k=top_k, top_p=top_p, silence=silence)
    elif 'bert' in model_path.lower():
        model = nml.Bert(model_path, device=device, temperature=temperature, top_k=top_k, top_p=top_p, silence=silence)
    elif 'xlnet' in model_path.lower():
        model = nml.XlNet(model_path, device=device, temperature=temperature, top_k=top_k, top_p=top_p, optimize=optimize,
            silence=silence)
    else:
        raise ValueError('Model name value is unexpected. Only support BERT, DistilBERT, RoBERTa and XLNet model.')

    CONTEXT_WORD_EMBS_MODELS[model_name] = model
    return model


class ContextualWordEmbsAug(WordAugmenter):
    # https://arxiv.org/pdf/1805.06201.pdf, https://arxiv.org/pdf/2003.02245.pdf
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
    :param str device: Default value is CPU. If value is CPU, it uses CPU for processing. If value is CUDA, it uses GPU
        for processing. Possible values include 'cuda' and 'cpu'. (May able to use other options)
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param bool optimize: If true, optimized process will be executed. For example, GPT2 will use "return_past" to
        reduce inference time.
    :param bool silence: Default is True. transformers library will print out warning message when leveraing
        pre-trained model. Set True to disable the expected warning message.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.ContextualWordEmbsAug()
    """

    def __init__(self, model_path='bert-base-uncased', action="substitute", temperature=1.0, top_k=100, top_p=None,
                 name='ContextualWordEmbs_Aug', aug_min=1, aug_max=10, aug_p=0.3, stopwords=None,
                 device='cpu', force_reload=False, optimize=None, stopwords_regex=None,
                 verbose=0, silence=True,):
        super().__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, tokenizer=None,
            device=device, stopwords=stopwords, verbose=verbose, stopwords_regex=stopwords_regex,
            include_detail=False, parallelable=True)
        self.model_path = model_path
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.silence = silence

        self._init()
        self.model = self.get_model(
            model_path=model_path, device=device, force_reload=force_reload, temperature=temperature, top_k=top_k,
            top_p=top_p, optimize=optimize, silence=silence)
        # Override stopwords
        if stopwords is not None and self.model_type in ['xlnet', 'roberta']:
            stopwords = [self.stopwords]
        self.device = self.model.device

        """
            TODO: Reserve 2 spaces (e.g. [CLS], [SEP]) is not enough as it hit CUDA error in batch processing mode.
            Therefore, forcing to reserve 5 times of reserved spaces (i.e. 5)
        """
        self.max_num_token = self.model.get_max_num_token()

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

    def is_stop_words(self, token):
        if self.model_type in ['bert', 'distilbert']:
            return super().is_stop_words(token)
        elif self.model_type in ['xlnet', 'roberta']:
            return self.stopwords is not None and token.replace(self.model.SUBWORD_PREFIX, '').lower() in self.stopwords
        return False

    def skip_aug(self, token_idxes, tokens):
        results = []

        for token_idx in token_idxes:
            token = tokens[token_idx]
            
            # Do not augment subword
            if self.model_type in ['bert', 'distilbert'] \
                and token.startswith(self.model.SUBWORD_PREFIX):
                continue
            # Do not augment tokens if len is less than aug_min
            if (self.model.SUBWORD_PREFIX in token and len(token) < self.aug_min+1) \
                or (self.model.SUBWORD_PREFIX not in token and len(token) < self.aug_min):
                continue
            if self.model_type in ['xlnet', 'roberta']:
                # xlent may tokenize word incorrectly. For example, 'fox', will be tokeinzed as ['_', 'fox']
                if token == self.model.SUBWORD_PREFIX:
                    continue

                # subword
                if not token.startswith(self.model.SUBWORD_PREFIX):
                    continue

            results.append(token_idx)

        return results

    def split_text(self, data):
        tokens = self.model.tokenizer.tokenize(data)

        if self.model.model.config.max_position_embeddings == -1:  # e.g. No max length restriction for XLNet
            return data, None, tokens, None  # Head text, tail text, head token, tail token

        ids = self.model.tokenizer.convert_tokens_to_ids(tokens[:self.max_num_token])
        head_text = self.model.tokenizer.decode(ids).strip()
        # head_text = self.model.tokenizer.convert_tokens_to_string(tokens[:self.max_num_token]).strip()
        tail_text = None
        if len(tokens) >= self.max_num_token:
            # tail_text = self.model.tokenizer.convert_tokens_to_string(tokens[self.max_num_token:]).strip()
            ids = self.model.tokenizer.convert_tokens_to_ids(tokens[self.max_num_token:])
            tail_text = self.model.tokenizer.decode(ids).strip()

        return head_text, tail_text, tokens[:self.max_num_token], tokens[self.max_num_token:]

    def insert(self, data):
        if not data:
            return data

        if isinstance(data, list):
            all_data = data
        else:
            if data.strip() == '':
                return data

            all_data = [data]

        # If length of input is larger than max allowed input, only augment heading part
        split_results = [self.split_text(d) for d in all_data] # head_text, tail_text, head_tokens, tail_tokens

        # Pick target word for augmentation
        for i, split_result in enumerate(split_results):
            head_text, tail_text, head_tokens, tail_tokens = split_result            

            if self.model_type in ['xlnet', 'roberta']:
                # xlent and roberta tokens include prefix (e.g. ▁ or Ġ')
                cleaned_head_tokens = [t.replace(self.model.SUBWORD_PREFIX, '') for t in head_tokens]
            else:
                cleaned_head_tokens = head_tokens

            head_doc = Doc(head_text, head_tokens)
            aug_idxes = self._get_aug_idxes(head_tokens)
            aug_idxes.sort(reverse=True)

            split_results[i] += (cleaned_head_tokens, head_doc, aug_idxes, )

        # Pad aug_idxes
        max_aug_size = max([len(split_result[6]) for split_result in split_results])
        for split_result in split_results:
            aug_idxes = split_result[6]
            for _ in range(max_aug_size - len(aug_idxes)):
                aug_idxes.append(-1)

        token_placeholder = self.model.MASK_TOKEN
        if self.model_type in ['xlnet', 'roberta']:
            token_placeholder = self.model.SUBWORD_PREFIX + token_placeholder  # Adding prefix for

        # Augment same index of aug by batch
        change_seq = 0
        for i in range(max_aug_size):
            masked_texts = []
            aug_input_poses = [] # store which input augmented. No record if padding

            change_seq += 1
            for j, split_result in enumerate(split_results):
                head_doc, aug_idx = split_result[5], split_result[6][i]

                # -1 if it is padding 
                if aug_idx == -1:
                    continue

                head_doc.add_token(aug_idx, token=token_placeholder, action=Action.INSERT,
                    change_seq=self.parent_change_seq+change_seq)

                aug_input_poses.append(j)
                # some tokenizers handle special charas (e.g. don't can merge after decode)
                if self.model_type in ['bert', 'distilbert']:
                    ids = self.model.tokenizer.convert_tokens_to_ids(head_doc.get_augmented_tokens())
                    masked_text = self.model.tokenizer.decode(ids).strip()
                elif self.model_type in ['xlnet', 'roberta']:
                    masked_text = self.model.tokenizer.convert_tokens_to_string(head_doc.get_augmented_tokens()).strip()

                masked_texts.append(masked_text)

            if not len(masked_texts):
                continue

            outputs = self.model.predict(masked_texts, target_words=None, n=2)

            # Update doc
            for aug_input_pos, output, masked_text in zip(aug_input_poses, outputs, masked_texts):
                split_result = split_results[aug_input_pos]
                head_doc = split_result[5]
                aug_idx = split_result[6][i] # augment position in text

                # TODO: Alternative method better than dropout
                candidate = ''
                if len(output) == 0:
                    # TODO: no result?
                    pass
                elif len(output) == 1:
                    candidate = output[0]
                elif len(output) > 1:
                    candidate = self.sample(output, 1)[0]

                # In XLNet, it can be the first word of sentence which does not come with sapce. E.g. Zombine (ID:29110)
                if self.model_type in ['xlnet', 'roberta']:
                    if candidate != '' and not candidate.startswith(self.model.SUBWORD_PREFIX):
                        candidate = self.model.SUBWORD_PREFIX + candidate

                head_doc.update_change_log(aug_idx, token=candidate)

                # Early stop if number of token exceed max number
                if head_doc.size() > self.max_num_token:
                    for j in range(i+1, max_aug_size):
                        split_results[aug_input_pos][6][j] = -1

        augmented_texts = []
        for split_result in split_results:
            tail_text, head_doc = split_result[1], split_result[5]

            head_tokens = head_doc.get_augmented_tokens()
            # if self.model_type in ['xlnet', 'roberta']:
            #     # xlent and roberta tokens include prefix (e.g. ▁ or Ġ')
            #     head_tokens = [self.model.SUBWORD_PREFIX + t if self.model.SUBWORD_PREFIX not in t and i != 0 else t for i, t in enumerate(head_tokens)]

            ids = self.model.tokenizer.convert_tokens_to_ids(head_tokens)
            augmented_text = self.model.tokenizer.decode(ids)
            if tail_text is not None:
                augmented_text += ' ' + tail_text
            augmented_texts.append(augmented_text)

        if isinstance(data, list):
            return augmented_texts
        else:
            return augmented_texts[0]

    def substitute(self, data):
        if not data:
            return data

        if isinstance(data, list):
            all_data = data
        else:
            if data.strip() == '':
                return data

            all_data = [data]

        # If length of input is larger than max allowed input, only augment heading part
        split_results = [self.split_text(d) for d in all_data] # head_text, tail_text, head_tokens, tail_tokens

        # Pick target word for augmentation
        for i, split_result in enumerate(split_results):
            head_text, tail_text, head_tokens, tail_tokens = split_result            

            if self.model_type in ['xlnet', 'roberta']:
                # xlent and roberta tokens include prefix (e.g. ▁ or Ġ')
                cleaned_head_tokens = [t.replace(self.model.SUBWORD_PREFIX, '') for t in head_tokens]
            else:
                cleaned_head_tokens = head_tokens

            head_doc = Doc(head_text, head_tokens)
            aug_idxes = self._get_aug_idxes(head_tokens)
            aug_idxes.sort(reverse=True)

            head_tokens = head_doc.get_augmented_tokens()

            split_results[i] += (cleaned_head_tokens, head_doc, aug_idxes, )

        # Pad aug_idxes
        max_aug_size = max([len(split_result[6]) for split_result in split_results])
        for split_result in split_results:
            aug_idxes = split_result[6]
            for _ in range(max_aug_size - len(aug_idxes)):
                aug_idxes.append(-1)

        token_placeholder = self.model.MASK_TOKEN
        if self.model_type in ['xlnet', 'roberta']:
            token_placeholder = self.model.SUBWORD_PREFIX + token_placeholder  # Adding prefix for

        # Augment same index of aug by batch
        change_seq = 0
        for i in range(max_aug_size):
            original_tokens = []
            masked_texts = []
            aug_input_poses = [] # store which input augmented. No record if padding

            change_seq += 1
            for j, split_result in enumerate(split_results):
                head_doc, aug_idx = split_result[5], split_result[6][i]

                # -1 if it is padding 
                if aug_idx == -1:
                    continue

                original_tokens.append(head_doc.get_token(aug_idx).get_latest_token().token)

                head_doc.add_change_log(aug_idx, new_token=token_placeholder, action=Action.SUBSTITUTE,
                    change_seq=self.parent_change_seq+change_seq)

                # remove continuous sub-word
                to_remove_idxes = []
                for k in range(aug_idx+1, head_doc.size()):
                    subword_token = head_doc.get_token(k).orig_token.token
                    if subword_token in string.punctuation:
                        break
                    if self.model_type in ['bert', 'distilbert'] and self.model.SUBWORD_PREFIX in subword_token:
                        to_remove_idxes.append(k)
                    elif self.model_type in ['xlnet', 'roberta'] and self.model.SUBWORD_PREFIX not in subword_token:
                        to_remove_idxes.append(k)
                    else:
                        break
                for k in reversed(to_remove_idxes):
                    head_doc.add_change_log(k, new_token='', action=Action.SUBSTITUTE,
                        change_seq=self.parent_change_seq+change_seq)

                aug_input_poses.append(j)
                # some tokenizers handle special charas (e.g. don't can merge after decode)
                if self.model_type in ['bert', 'distilbert']:
                    ids = self.model.tokenizer.convert_tokens_to_ids(head_doc.get_augmented_tokens())
                    masked_text = self.model.tokenizer.decode(ids).strip()
                elif self.model_type in ['xlnet', 'roberta']:
                    masked_text = self.model.tokenizer.convert_tokens_to_string(head_doc.get_augmented_tokens()).strip()
                masked_texts.append(masked_text)

            if not len(masked_texts):
                continue

            outputs = self.model.predict(masked_texts, target_words=original_tokens, n=2)

            # Update doc
            for original_token, aug_input_pos, output, masked_text in zip(original_tokens, aug_input_poses, outputs, masked_texts):
                split_result = split_results[aug_input_pos]
                head_doc = split_result[5]
                aug_idx = split_result[6][i] # augment position in text

                # TODO: Alternative method better than dropout
                candidate = ''
                if len(output) == 0:
                    # TODO: no result?
                    pass
                elif len(output) == 1:
                    candidate = output[0]
                elif len(output) > 1:
                    candidate = self.sample(output, 1)[0]

                # In XLNet, it can be the first word of sentence which does not come with sapce. E.g. Zombine (ID:29110)
                if self.model_type in ['xlnet', 'roberta']:
                    if candidate != '' and not candidate.startswith(self.model.SUBWORD_PREFIX):
                        candidate = self.model.SUBWORD_PREFIX + candidate

                # Fallback to original token if no candidate is appropriate
                if candidate == '':
                    candidate = original_token

                head_doc.update_change_log(aug_idx, token=candidate, action=Action.SUBSTITUTE,
                    change_seq=self.parent_change_seq+change_seq)

                # Early stop if number of token exceed max number
                if head_doc.size() > self.max_num_token:
                    for j in range(i+1, max_aug_size):
                        split_results[aug_input_pos][6][j] = -1

        augmented_texts = []
        for split_result in split_results:
            tail_text, head_doc = split_result[1], split_result[5]

            head_tokens = head_doc.get_augmented_tokens()
            # if self.model_type in ['xlnet', 'roberta']:
            #     # xlent and roberta tokens include prefix (e.g. ▁ or Ġ')
            #     head_tokens = [self.model.SUBWORD_PREFIX + t if self.model.SUBWORD_PREFIX not in t and i != 0 else t for i, t in enumerate(head_tokens)]

            ids = self.model.tokenizer.convert_tokens_to_ids(head_tokens)
            augmented_text = self.model.tokenizer.decode(ids)
            if tail_text is not None:
                augmented_text += ' ' + tail_text
            augmented_texts.append(augmented_text)

        if isinstance(data, list):
            return augmented_texts
        else:
            return augmented_texts[0]

    @classmethod
    def get_model(cls, model_path, device='cuda', force_reload=False, temperature=1.0, top_k=None, top_p=0.0,
                  optimize=None, silence=True):
        return init_context_word_embs_model(model_path, device, force_reload, temperature, top_k, top_p, optimize, silence)
