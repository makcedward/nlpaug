try:
    import torch
    from transformers import RobertaTokenizer, RobertaForMaskedLM
    # from transformers import AutoModel, AutoTokenizer
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels
from nlpaug.util.selection.filtering import *


class Roberta(LanguageModels):
    # https://arxiv.org/pdf/1810.04805.pdf
    START_TOKEN = '<s>'
    SEPARATOR_TOKEN = '</s>'
    MASK_TOKEN = '<mask>'
    SUBWORD_PREFIX = 'Ġ'

    def __init__(self, model_path='roberta-base', temperature=1.0, top_k=None, top_p=None, device='cuda'):
        super().__init__(device, temperature=temperature, top_k=top_k, top_p=top_p)
        self.model_path = model_path

        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForMaskedLM.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

    def id2token(self, _id):
        # id: integer format
        return self.tokenizer.convert_ids_to_tokens([_id])[0]

    def is_skip_candidate(self, candidate):
        return False

    def predict(self, text, target_word=None, n=1):
        # Prepare inputs
        tokens = self.tokenizer.tokenize(text)

        tokens.insert(0, self.START_TOKEN)
        tokens.append(self.SEPARATOR_TOKEN)
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
            outputs = self.model(input_ids=token_inputs, token_type_ids=segment_inputs, attention_mask=mask_inputs)
        target_token_logits = outputs[0][0][target_pos]

        # Selection
        seed = {'temperature': self.temperature, 'top_k': self.top_k, 'top_p': self.top_p}
        target_token_logits = self.control_randomness(target_token_logits, seed)
        target_token_logits, target_token_idxes = self.filtering(target_token_logits, seed)
        if len(target_token_idxes) != 0:
            results = self.pick(target_token_logits, target_token_idxes, target_word=target_word, n=n)
            # Replace '</s>' and 'Ġ' as . and empty string
            results = [(r[0].replace('Ġ', ''), r[1]) if r[0] != self.SEPARATOR_TOKEN else ('.', r[1]) for r in results]
        else:
            results = None

        results = (results,)

        return results
