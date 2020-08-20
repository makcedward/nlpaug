try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels

import nlpaug.util.text.tokenizer as text_tokenizer


class Bart(LanguageModels):
    # https://arxiv.org/pdf/1910.13461.pdf
    def __init__(self, model_path='facebook/bart-large-cnn', min_length=10, max_length=20, num_beam=3, no_repeat_ngram_size=3, device='cuda'):
        super().__init__(device, temperature=None, top_k=None, top_p=None)
        try:
            import transformers
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed transformers library. Install transfomers by `pip install transformers`')

        self.model_path = model_path
        self.min_length = min_length
        self.max_length = max_length
        self.num_beam = num_beam
        self.no_repeat_ngram_size = no_repeat_ngram_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

        self.return_tensor = 'pt' # PyTorch
        self.early_stopping = True
        self.skip_special_token = True
        self.default_max_length_ratio = 0.5

    def predict(self, text, n=1):
        # Convert to feature
        token_ids = self.tokenizer.encode(text, return_tensors=self.return_tensor)

        # Prediction
        min_length = self.get_min_length(text)
        max_length = self.get_max_length(text)
        target_token_ids = self.model.generate(token_ids,
            min_length=min_length, max_length=max_length, num_beams=self.num_beam,
            no_repeat_ngram_size=self.no_repeat_ngram_size)

        tokens = self.tokenizer.decode(target_token_ids[0], skip_special_tokens=self.skip_special_token)

        # Return full sentence only.
        for i in range(len(tokens)-1, -1, -1):
            if tokens[i] in text_tokenizer.SENTENCE_SEPARATOR:
                return tokens[:i+1]

        return tokens

    def get_min_length(self, text):
        return int(len(text) * self.min_length) if self.min_length < 1 else self.min_length

    def get_max_length(self, text):
        if self.max_length < 1:
            return int(len(text) * self.max_length)
        else:
            if len(text) >= self.max_length:
                return int(len(text) * self.default_max_length_ratio)
            else:
                return self.max_length
