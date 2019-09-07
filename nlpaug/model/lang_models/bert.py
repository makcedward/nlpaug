# Source: https://arxiv.org/abs/1810.04805

try:
    import torch
    from pytorch_transformers import BertTokenizer, BertForMaskedLM
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels
from nlpaug.util.selection.filtering import *


class BertDeprecated(LanguageModels):
    START = '[CLS]'
    SEPARATOR = '[SEP]'
    MASK = '[MASK]'
    SUBWORD_PREFIX = '##'

    def __init__(self, model_path='bert-base-uncased', tokenizer_path=None, device='cuda'):
        super().__init__(device)
        self.model_path = model_path

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForMaskedLM.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    def predict(self, input_tokens, target_word, top_n):
        results = []

        tokens = input_tokens.copy()
        tokens.insert(0, BertDeprecated.START)
        tokens.append(BertDeprecated.SEPARATOR)

        # Mask target word
        target_pos = tokens.index(target_word)
        target_idx = self.tokenizer.convert_tokens_to_ids([target_word])[0]
        tokens[target_pos] = BertDeprecated.MASK

        # Convert feature
        token_idxes = self.tokenizer.convert_tokens_to_ids(tokens)
        segments_idxes = [0] * len(tokens)
        mask_idxes = [1] * len(tokens)  # 1: real token, 0: padding token

        tokens_features = torch.tensor([token_idxes]).to(self.device)
        segments_features = torch.tensor([segments_idxes]).to(self.device)
        mask_features = torch.tensor([mask_idxes]).to(self.device)

        # Predict target word
        outputs = self.model(tokens_features, segments_features, mask_features)
        logits = outputs[0]

        top_score_idx = target_idx
        for _ in range(100):
            logits[0, target_pos, top_score_idx] = -9999
            top_score_idx = torch.argmax(logits[0, target_pos]).item()
            top_score_token = self.tokenizer.convert_ids_to_tokens([top_score_idx])[0]
            if top_score_token[:2] != Bert.SUBWORD_PREFIX:
                results.append(top_score_token)
                if len(results) >= top_n:
                    break

        return results


class Bert(LanguageModels):
    START_TOKEN = '[CLS]'
    SEPARATOR_TOKEN = '[SEP]'
    MASK_TOKEN = '[MASK]'
    SUBWORD_PREFIX = '##'

    def __init__(self, model_path='bert-base-cased', device='cuda'):
        super().__init__(device)
        self.model_path = model_path

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForMaskedLM.from_pretrained(model_path)

        self.model.to(device)
        self.model.eval()

    def id2token(self, _id):
        # id: integer format
        return self.tokenizer.convert_ids_to_tokens([_id])[0]

    def is_skip_candidate(self, candidate):
        return candidate[:2] == self.SUBWORD_PREFIX

    def predict(self, text, target_word=None, top_n=5):
        # Prepare inputs
        tokens = self.tokenizer.tokenize(text)
        tokens.insert(0, Bert.START_TOKEN)
        tokens.append(Bert.SEPARATOR_TOKEN)
        target_pos = tokens.index(self.MASK_TOKEN)

        token_inputs = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_inputs = [0] * len(token_inputs)
        mask_inputs = [1] * len(token_inputs)  # 1: real token, 0: padding token

        # Convert to feature
        token_inputs = torch.tensor([token_inputs]).to(self.device)
        segment_inputs = torch.tensor([segment_inputs]).to(self.device)
        mask_inputs = torch.tensor([mask_inputs]).to(self.device)

        # Prediction
        with torch.no_grad():
            outputs = self.model(token_inputs, segment_inputs, mask_inputs)
        target_token_logits = outputs[0][0][target_pos]

        # Filtering
        if self.top_k > 0:
            target_token_logits, target_token_idxes = filter_top_n(
                target_token_logits, top_n + self.top_k, -float('Inf'))

        # Generate candidates
        candidate_ids, candidate_probas = self.prob_multinomial(target_token_logits, top_n=top_n + 10)
        results = self.get_candidiates(candidate_ids, candidate_probas, target_word, top_n)
        return results
