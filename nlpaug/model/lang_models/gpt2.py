try:
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    # from transformers import AutoModel, AutoTokenizer
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels


class Gpt2(LanguageModels):
    # https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    SUBWORD_PREFIX = 'Ġ'

    def __init__(self, model_path='gpt2', temperature=1.0, top_k=None, top_p=None, device=None, optimize=None):
        super().__init__(device, temperature=temperature, top_k=top_k, top_p=top_p, optimize=optimize)
        self.model_path = model_path

        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

    def id2token(self, _id):
        return self.tokenizer.decode(_id, clean_up_tokenization_spaces=True).strip()

    def predict(self, text, target_word=None, n=1, external_memory=None):
        # Convert feature
        input_idxes = self.tokenizer.encode(text)
        # if self.optimize['external_memory']:
        #     input_idxes = input_idxes[-1:]
        input_idxes = torch.tensor(input_idxes, device=self.device).unsqueeze(0).repeat(1, 1)

        # Prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_idxes, past=external_memory)
        target_token_logits = outputs[0][0][-1]  # GPT2 only predict last token

        # Selection
        seed = {'temperature': self.temperature, 'top_k': self.top_k, 'top_p': self.top_p}
        target_token_logits = self.control_randomness(target_token_logits, seed)
        target_token_logits, target_token_idxes = self.filtering(target_token_logits, seed)
        if len(target_token_idxes) != 0:
            results = self.pick(target_token_logits, target_token_idxes, target_word=target_word, n=n)
        else:
            results = None

        results = (results,)
        if self.optimize['external_memory']:
            external_memory = outputs[1]
            results += (external_memory,)

        return results
