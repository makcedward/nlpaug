try:
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels


class Gpt2(LanguageModels):
    # https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    SUBWORD_PREFIX = 'Ġ'

    def __init__(self, model_path='gpt2', temperature=1.0, top_k=None, top_p=None, device=None, return_past=False):
        super().__init__(device, temperature=temperature, top_k=top_k, top_p=top_p)
        self.model_path = model_path

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

        self.return_past = return_past

    def id2token(self, _id):
        return self.tokenizer.decode(_id, clean_up_tokenization_spaces=True).strip()

    def predict(self, text, target_word=None, n=1, past=None):
        # Convert feature
        input_idxes = self.tokenizer.encode(text)
        if past is not None:
            input_idxes = input_idxes[-1:]
        input_idxes = torch.tensor(input_idxes, device=self.device).unsqueeze(0).repeat(1, 1)

        # Prediction
        with torch.no_grad():
            prediction_scores, past = self.model(input_ids=input_idxes, past=past)
        target_token_logits = prediction_scores[0][-1]  # GPT2 only predict last token

        # Selection
        seed = {'temperature': self.temperature, 'top_k': self.top_k, 'top_p': self.top_p}
        target_token_logits = self.control_randomness(target_token_logits, seed)
        target_token_logits, target_token_idxes = self.filtering(target_token_logits, seed)

        results = self.pick(target_token_logits, target_word=target_word, n=n)
        if self.return_past:
            return results, past
        else:
            return results
