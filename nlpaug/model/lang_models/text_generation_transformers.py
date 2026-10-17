try:
    import torch
    from transformers import pipeline
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels


class TextGenTransformers(LanguageModels):
    def __init__(self, model_path='gpt2', device='cuda', min_length=100, max_length=300, 
        batch_size=32, temperature=1.0, top_k=50, top_p=0.9, silence=True):
        super().__init__(device, model_type=None, silence=silence)
        try:
            from transformers import pipeline
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed transformers library. Install transfomers by `pip install transformers`')

        self.min_length = min_length
        self.max_length = max_length
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.model_path = model_path
        self.device = self.convert_device(device)

        self.model = self._load_with_optional_silence(
            lambda: pipeline("text-generation", model=model_path, device=self.device),
            silence=silence,
        )

    def to(self, device):
        self.model.model.to(device)

    def get_device(self):
        return str(self.model.device)

    def _warn_if_input_exceeds_max_length(self, texts):
        # Issue #285: length is counted after tokenization, so it exceeds
        # len(text.split()). Warn early with actionable guidance instead of
        # leaving the raw transformers truncation warning unexplained.
        try:
            tokenizer = self.model.tokenizer
        except AttributeError:
            return
        try:
            for text in texts:
                token_len = len(tokenizer.encode(text))
                if token_len > (self.max_length or 0):
                    import warnings
                    warnings.warn(
                        f'Input length {token_len} (tokenized) exceeds max_length={self.max_length}. '
                        'Length is measured after tokenization, not len(text.split()). '
                        'Increase max_length/config.max_length or split the document '
                        '(see split_long_document, issue #336). (issue #285)')
                    break
        except Exception:
            pass

    def predict(self, texts, target_words=None, n=1):
        self._warn_if_input_exceeds_max_length(texts)
        results = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                predict_result = self.model(
                    texts[i:i+self.batch_size], 
                    pad_token_id=50256,
                    min_length=self.min_length, 
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    do_sample=True,
                    num_return_sequences=1,
                    # Issue #330: avoid DataLoader worker crash on macOS.
                    num_workers=0
                )
                if isinstance(predict_result, list):
                    results.extend([y for x in predict_result for y in x])

        return [r['generated_text'] for r in results]
