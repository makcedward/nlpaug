# Source: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

try:
    import torch
    from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels
from nlpaug.util.selection.filtering import *


class Gpt2(LanguageModels):
    SUBWORD_PREFIX = 'Ġ'

    def __init__(self, model_path='gpt2', top_k=None, top_p=None, device=None):
        super().__init__(device, top_k=top_k, top_p=top_p)
        self.model_path = model_path

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    def id2token(self, _id):
        return self.tokenizer.decode(_id, clean_up_tokenization_spaces=True).strip()

    def predict(self, text, target_word=None, top_n=5):
        # Convert feature
        input_idxes = self.tokenizer.encode(text)
        input_idxes = torch.tensor(input_idxes, device=self.device).unsqueeze(0).repeat(1, 1)

        # Prediction
        with torch.no_grad():
            outputs = self.model(input_idxes)
        target_token_logits = outputs[0][0][-1]  # GPT2 only predict last token

        # Filtering
        if self.top_k is not None and 0 < self.top_k < len(target_token_logits):
            target_token_logits, target_token_idxes = filter_top_n(
                target_token_logits, top_n + self.top_k, -float('Inf'))
        if self.top_p is not None and 0 < self.top_p < 1:
            target_token_logits, target_token_idxes = filter_cum_proba(target_token_logits, self.top_p)

        # Generate candidates
        candidate_ids, candidate_probas = self.prob_multinomial(target_token_logits, top_n=top_n + 10)
        results = self.get_candidiates(candidate_ids, candidate_probas, target_word, top_n)
        return results
