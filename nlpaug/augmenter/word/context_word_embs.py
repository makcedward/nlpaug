"""
    Augmenter that apply operation (word level) to textual input based on contextual word embeddings.
"""

import string
import os
import re
import logging

from nlpaug.augmenter.word import WordAugmenter
import nlpaug.model.lang_models as nml
from nlpaug.util import Action, Doc

CONTEXT_WORD_EMBS_MODELS = {}


def init_context_word_embs_model(model_path, model_type, device, force_reload=False, batch_size=32, 
    top_k=None, silence=True, use_custom_api=False):
    global CONTEXT_WORD_EMBS_MODELS

    model_name = '_'.join([os.path.basename(model_path), model_type, str(device)])
    if model_name in CONTEXT_WORD_EMBS_MODELS and not force_reload:
        CONTEXT_WORD_EMBS_MODELS[model_name].top_k = top_k
        CONTEXT_WORD_EMBS_MODELS[model_name].batch_size = batch_size
        CONTEXT_WORD_EMBS_MODELS[model_name].silence = silence
        return CONTEXT_WORD_EMBS_MODELS[model_name]

    if use_custom_api:
        if model_type == 'distilbert':
            model = nml.DistilBert(model_path, device=device, top_k=top_k, silence=silence, batch_size=batch_size)
        elif model_type == 'roberta':
            model = nml.Roberta(model_path, device=device, top_k=top_k, silence=silence, batch_size=batch_size)
        elif model_type == 'bert':
            model = nml.Bert(model_path, device=device, top_k=top_k, silence=silence, batch_size=batch_size)
        else:
            raise ValueError('Model type value is unexpected. Only support bert and roberta models.')
    else:
        if model_type in ['distilbert', 'bert', 'roberta', 'bart']:
            model = nml.FmTransformers(model_path, model_type=model_type, device=device, batch_size=batch_size,
                top_k=top_k, silence=silence)
        else:
            raise ValueError('Model type value is unexpected. Only support bert and roberta models.')

    CONTEXT_WORD_EMBS_MODELS[model_name] = model
    return model

class ContextualWordEmbsAug(WordAugmenter):
    # https://arxiv.org/pdf/1805.06201.pdf, https://arxiv.org/pdf/2003.02245.pdf
    """
    Augmenter that leverage contextual word embeddings to find top n similar word for augmentation.

    :param str model_path: Model name or model path. It used transformers to load the model. Tested
        'bert-base-uncased', 'bert-base-cased', 'distilbert-base-uncased', 'roberta-base', 'distilroberta-base',
        'facebook/bart-base', 'squeezebert/squeezebert-uncased'.
    :param str model_type: Type of model. For BERT model, use 'bert'. For RoBERTa/LongFormer model, use 'roberta'. 
        For BART model, use 'bart'. If no value is provided, will determine from model name.
    :param str action: Either 'insert or 'substitute'. If value is 'insert', a new word will be injected to random
        position according to contextual word embeddings calculation. If value is 'substitute', word will be replaced
        according to contextual embeddings calculation
    :param int top_k: Controlling lucky draw pool. Top k score token will be used for augmentation. Larger k, more
        token can be used. Default value is 100. If value is None which means using all possible tokens.
    :param float aug_p: Percentage of word will be augmented.
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result from
        aug_p. Otherwise, using aug_max.
    :param list stopwords: List of words which will be skipped from augment operation. Do NOT include the UNKNOWN word.
        UNKNOWN word of BERT is [UNK]. UNKNOWN word of RoBERTa and BART is <unk>.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param str device: Default value is CPU. If value is CPU, it uses CPU for processing. If value is CUDA, it uses GPU
        for processing. Possible values include 'cuda' and 'cpu'. (May able to use other options)
    :param int batch_size: Batch size.
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param bool silence: Default is True. transformers library will print out warning message when leveraing
        pre-trained model. Set True to disable the expected warning message.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.ContextualWordEmbsAug()
    """

    def __init__(self, model_path='bert-base-uncased', model_type='', action="substitute", top_k=100, 
                 name='ContextualWordEmbs_Aug', aug_min=1, aug_max=10, aug_p=0.3, stopwords=None,
                 batch_size=32, device='cpu', force_reload=False, stopwords_regex=None,
                 verbose=0, silence=True, use_custom_api=True):
        super().__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, tokenizer=None,
            device=device, stopwords=stopwords, verbose=verbose, stopwords_regex=stopwords_regex,
            include_detail=False)
        self.model_path = model_path
        self.model_type = model_type if model_type != '' else self.check_model_type() 
        self.silence = silence

        # TODO: Slow when switching to HuggingFace pipeline. #https://github.com/makcedward/nlpaug/issues/248
        self.use_custom_api = use_custom_api

        self.model = self.get_model(
            model_path=model_path, model_type=self.model_type, device=device, force_reload=force_reload,
            batch_size=batch_size, top_k=top_k, silence=silence, use_custom_api=use_custom_api)
        # Override stopwords
        # if stopwords and self.model_type in ['xlnet', 'roberta']:
        #     stopwords = [self.stopwords]

        # lower case all stopwords
        if stopwords and 'uncased' in model_path:
            self.stopwords = [s.lower() for s in self.stopwords]

        self.stopword_reg = None
        self.reserve_word_reg = None
        self._build_stop_words(stopwords)

        self.device = self.model.device

        """
            TODO: Reserve 2 spaces (e.g. [CLS], [SEP]) is not enough as it hit CUDA error in batch processing mode.
            Therefore, forcing to reserve 5 times of reserved spaces (i.e. 5)
        """
        self.max_num_token = self.model.get_max_num_token()

    def _build_stop_words(self, stopwords):
        if stopwords:
            prefix_reg = '(?<=\s|\W)'
            suffix_reg = '(?=\s|\W)'
            stopword_reg = '('+')|('.join([prefix_reg + re.escape(s) + suffix_reg for s in stopwords])+')'
            self.stopword_reg = re.compile(stopword_reg)

            unknown_token = self.model.get_unknown_token() or self.model.UNKNOWN_TOKEN
            reserve_word_reg = '(' + prefix_reg + re.escape(unknown_token) + suffix_reg + ')'
            self.reserve_word_reg = re.compile(reserve_word_reg)

    def check_model_type(self):
        # if 'xlnet' in self.model_path.lower():
        #     return 'xlnet'

        if 'longformer' in self.model_path.lower():
            return 'roberta' 
        elif 'roberta' in self.model_path.lower():
            return 'roberta'

        elif 'distilbert' in self.model_path.lower():
            return 'bert'
        elif 'squeezebert' in self.model_path.lower():
            return 'bert'
        elif 'bert' in self.model_path.lower():
            return 'bert'

        elif 'bart' in self.model_path.lower():
            return 'bart'

#     'google/electra-small-discriminator',
#     'google/reformer-enwik8',
#     'funnel-transformer/small-base',
#     'google/tapas-base',
#     'microsoft/deberta-base'
        
        return ''

    def is_stop_words(self, token):
        # Will execute before any tokenization. No need to handle prefix processing
        if self.stopwords:
            unknown_token = self.model.get_unknown_token() or self.model.UNKNOWN_TOKEN
            if token == unknown_token:
                return True
            return token.lower() in self.stopwords
        else:
            return False

    def skip_aug(self, token_idxes, tokens):
        results = []

        for token_idx in token_idxes:
            token = tokens[token_idx]

            # Do not augment subword
            if self.model_type in ['bert', 'electra'] \
                and token.startswith(self.model.get_subword_prefix()):
                continue
            # Do not augment tokens if len is less than aug_min
            if (self.model.get_subword_prefix() in token and len(token) < self.aug_min+1) \
                or (self.model.get_subword_prefix() not in token and len(token) < self.aug_min):
                continue
            if self.model_type in ['xlnet', 'roberta', 'bart']:
                # xlent may tokenize word incorrectly. For example, 'fox', will be tokeinzed as ['_', 'fox']
                if token == self.model.get_subword_prefix():
                    continue

                # subword
                if not token.startswith(self.model.get_subword_prefix()):
                    continue

            results.append(token_idx)

        return results

    def split_text(self, data):
        # Expect to have waring for "Token indices sequence length is longer than the specified maximum sequence length for this model"

        # Handle stopwords first #https://github.com/makcedward/nlpaug/issues/247
        if self.stopwords:
            unknown_token = self.model.get_unknown_token() or self.model.UNKNOWN_TOKEN
            preprocessed_data, reserved_stopwords = self.replace_stopword_by_reserved_word(data, self.stopword_reg, unknown_token)
        else:
            preprocessed_data, reserved_stopwords = data, None

        orig_log_level = logging.getLogger('transformers.' + 'tokenization_utils_base').getEffectiveLevel()
        logging.getLogger('transformers.' + 'tokenization_utils_base').setLevel(logging.ERROR)
        tokens = self.model.get_tokenizer().tokenize(preprocessed_data)
        logging.getLogger('transformers.' + 'tokenization_utils_base').setLevel(orig_log_level)

        if self.model.get_model().config.max_position_embeddings == -1:  # e.g. No max length restriction for XLNet
            return (preprocessed_data, None, tokens, None), reserved_stopwords  # (Head text, tail text, head token, tail token), reserved_stopwords

        ids = self.model.get_tokenizer().convert_tokens_to_ids(tokens[:self.max_num_token])
        head_text = self.model.get_tokenizer().decode(ids).strip()
        # head_text = self.model.get_tokenizer().convert_tokens_to_string(tokens[:self.max_num_token]).strip()
        tail_text = None
        if len(tokens) >= self.max_num_token:
            # tail_text = self.model.get_tokenizer().convert_tokens_to_string(tokens[self.max_num_token:]).strip()
            ids = self.model.get_tokenizer().convert_tokens_to_ids(tokens[self.max_num_token:])
            tail_text = self.model.get_tokenizer().decode(ids).strip()

        return (head_text, tail_text, tokens[:self.max_num_token], tokens[self.max_num_token:]), reserved_stopwords

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
        split_results = [] # head_text, tail_text, head_tokens, tail_tokens
        reserved_stopwords = []
        for d in all_data:
            split_result, reserved_stopword = self.split_text(d)
            split_results.append(split_result)
            reserved_stopwords.append(reserved_stopword)

        change_seq = 0

        # Pick target word for augmentation
        for i, (split_result, reserved_stopword_tokens) in enumerate(zip(split_results, reserved_stopwords)):
            head_text, tail_text, head_tokens, tail_tokens = split_result            

            if self.model_type in ['xlnet', 'roberta', 'bart']:
                # xlent and roberta tokens include prefix (e.g. ▁ or Ġ')
                cleaned_head_tokens = [t.replace(self.model.get_subword_prefix(), '') for t in head_tokens]
            else:
                cleaned_head_tokens = head_tokens

            head_doc = Doc(head_text, head_tokens)
            aug_idxes = self._get_aug_idxes(head_tokens)
            aug_idxes.sort(reverse=True)
            if reserved_stopword_tokens:
                head_doc, change_seq = self.substitute_back_reserved_stopwords(
                    head_doc, reserved_stopword_tokens, change_seq)

            split_results[i] += (cleaned_head_tokens, head_doc, aug_idxes, )

        # Pad aug_idxes
        max_aug_size = max([len(split_result[6]) for split_result in split_results])
        for split_result in split_results:
            aug_idxes = split_result[6]
            for _ in range(max_aug_size - len(aug_idxes)):
                aug_idxes.append(-1)

        token_placeholder = self.model.get_mask_token()
        if self.model_type in ['xlnet', 'roberta', 'bart']:
            token_placeholder = self.model.get_subword_prefix() + token_placeholder  # Adding prefix for

        # Augment same index of aug by batch
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
                if self.model_type in ['bert', 'electra']:
                    ids = self.model.get_tokenizer().convert_tokens_to_ids(head_doc.get_augmented_tokens())
                    masked_text = self.model.get_tokenizer().decode(ids).strip()
                elif self.model_type in ['xlnet', 'roberta', 'bart']:
                    masked_text = self.model.get_tokenizer().convert_tokens_to_string(head_doc.get_augmented_tokens()).strip()

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

                # # In XLNet, it can be the first word of sentence which does not come with space. E.g. Zombine (ID:29110)
                # if self.model_type in ['xlnet']:
                #     if candidate != '' and not candidate.startswith(self.model.get_subword_prefix()):
                #         candidate = self.model.get_subword_prefix() + candidate
                # if self.model_type in ['roberta', 'bart']:
                #     if candidate != '' and not candidate.startswith(self.model.get_subword_prefix()) and candidate.strip() != candidate:
                #         candidate = self.model.get_subword_prefix() + candidate.strip()

                # no candidate
                if candidate == '':
                    head_doc.add_change_log(aug_idx, new_token='', action=Action.DELETE, change_seq=self.parent_change_seq+change_seq)
                    continue

                head_doc.update_change_log(aug_idx, token=candidate)

                # Early stop if number of token exceed max number
                if head_doc.size() > self.max_num_token:
                    for j in range(i+1, max_aug_size):
                        split_results[aug_input_pos][6][j] = -1

        augmented_texts = []
        for split_result, reserved_stopword_tokens in zip(split_results, reserved_stopwords):
            tail_text, head_doc = split_result[1], split_result[5]

            head_tokens = head_doc.get_augmented_tokens()
            # if self.model_type in ['xlnet', 'roberta']:
            #     # xlent and roberta tokens include prefix (e.g. ▁ or Ġ')
            #     head_tokens = [self.model.get_subword_prefix() + t if self.model.get_subword_prefix() not in t and i != 0 else t for i, t in enumerate(head_tokens)]

            ids = self.model.get_tokenizer().convert_tokens_to_ids(head_tokens)
            augmented_text = self.model.get_tokenizer().decode(ids)

            if tail_text:
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
        split_results = [] # head_text, tail_text, head_tokens, tail_tokens
        reserved_stopwords = []
        for d in all_data:
            split_result, reserved_stopword = self.split_text(d)
            split_results.append(split_result)
            reserved_stopwords.append(reserved_stopword)

        change_seq = 0
        # Pick target word for augmentation
        for i, (split_result, reserved_stopword_tokens) in enumerate(zip(split_results, reserved_stopwords)):
            head_text, tail_text, head_tokens, tail_tokens = split_result            

            if self.model_type in ['xlnet', 'roberta', 'bart']:
                # xlent and roberta tokens include prefix (e.g. ▁ or Ġ')
                cleaned_head_tokens = [t.replace(self.model.get_subword_prefix(), '') for t in head_tokens]
            else:
                cleaned_head_tokens = head_tokens

            head_doc = Doc(head_text, head_tokens)
            aug_idxes = self._get_aug_idxes(head_tokens)
            aug_idxes.sort(reverse=True)

            if reserved_stopword_tokens:
                head_doc, change_seq = self.substitute_back_reserved_stopwords(
                    head_doc, reserved_stopword_tokens, change_seq)
            head_tokens = head_doc.get_augmented_tokens()
            

            split_results[i] += (cleaned_head_tokens, head_doc, aug_idxes, )

        # Pad aug_idxes
        max_aug_size = max([len(split_result[6]) for split_result in split_results])
        for split_result in split_results:
            aug_idxes = split_result[6]
            for _ in range(max_aug_size - len(aug_idxes)):
                aug_idxes.append(-1)

        token_placeholder = self.model.get_mask_token()
        if self.model_type in ['xlnet', 'roberta', 'bart']:
            token_placeholder = self.model.get_subword_prefix() + token_placeholder  # Adding prefix for

        # Augment same index of aug by batch
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
                    if self.model_type in ['bert', 'electra'] and self.model.get_subword_prefix() in subword_token:
                        to_remove_idxes.append(k)
                    elif self.model_type in ['xlnet', 'roberta', 'bart'] and self.model.get_subword_prefix() not in subword_token:
                        to_remove_idxes.append(k)
                    else:
                        break
                for k in reversed(to_remove_idxes):
                    head_doc.add_change_log(k, new_token='', action=Action.SUBSTITUTE,
                        change_seq=self.parent_change_seq+change_seq)

                aug_input_poses.append(j)

                # some tokenizers handle special charas (e.g. don't can merge after decode)
                if self.model_type in ['bert', 'electra']:
                    ids = self.model.get_tokenizer().convert_tokens_to_ids(head_doc.get_augmented_tokens())
                    masked_text = self.model.get_tokenizer().decode(ids).strip()
                elif self.model_type in ['xlnet', 'roberta', 'bart']:
                    masked_text = self.model.get_tokenizer().convert_tokens_to_string(head_doc.get_augmented_tokens()).strip()

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

                # # In XLNet, it can be the first word of sentence which does not come with space. E.g. Zombine (ID:29110)
                # if self.model_type in ['xlnet']:
                #     if candidate != '' and not candidate.startswith(self.model.get_subword_prefix()):
                #         candidate = self.model.get_subword_prefix() + candidate
                # if self.model_type in ['roberta', 'bart']:
                #     if candidate != '' and not candidate.startswith(self.model.get_subword_prefix()) and candidate.strip() != candidate:
                #         candidate = self.model.get_subword_prefix() + candidate.strip()

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
            #     head_tokens = [self.model.get_subword_prefix() + t if self.model.get_subword_prefix() not in t and i != 0 else t for i, t in enumerate(head_tokens)]

            ids = self.model.get_tokenizer().convert_tokens_to_ids(head_tokens)
            augmented_text = self.model.get_tokenizer().decode(ids)

            if tail_text is not None:
                augmented_text += ' ' + tail_text
            augmented_texts.append(augmented_text)

        if isinstance(data, list):
            return augmented_texts
        else:
            return augmented_texts[0]

    @classmethod
    def get_model(cls, model_path, model_type, device='cuda', force_reload=False, batch_size=32,
        top_k=None, silence=True, use_custom_api=False):
        return init_context_word_embs_model(model_path, model_type, device, force_reload, batch_size, top_k,
            silence, use_custom_api)

    def substitute_back_reserved_stopwords(self, doc, reserved_stopword_tokens, change_seq):
        unknown_token = self.model.get_unknown_token() or self.model.UNKNOWN_TOKEN
        reserved_pos = len(reserved_stopword_tokens) - 1
        for token_i, token in enumerate(doc.get_augmented_tokens()):
            if token == unknown_token:
                change_seq += 1
                doc.update_change_log(token_i, token=reserved_stopword_tokens[reserved_pos], 
                    action=Action.SUBSTITUTE,
                    change_seq=self.parent_change_seq+change_seq)
                reserved_pos -= 1
        return doc, change_seq
