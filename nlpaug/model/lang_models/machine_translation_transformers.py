try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels


class MtTransformers(LanguageModels):
    def __init__(self, src_model_name='facebook/wmt19-en-de', tgt_model_name='facebook/wmt19-de-en',
                 device='cuda', silence=True, batch_size=32, max_length=None):
        super().__init__(device, model_type=None, silence=silence)
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed transformers library. Install transfomers by `pip install transformers`')

        self.src_model_name = src_model_name
        self.tgt_model_name = tgt_model_name
        self.src_model = self._load_with_optional_silence(
            lambda: AutoModelForSeq2SeqLM.from_pretrained(self.src_model_name),
            silence=silence,
        )
        self.src_model.eval()
        self.src_model.to(device)
        self.tgt_model = self._load_with_optional_silence(
            lambda: AutoModelForSeq2SeqLM.from_pretrained(self.tgt_model_name),
            silence=silence,
        )
        self.tgt_model.eval()
        self.tgt_model.to(device)
        self.src_tokenizer = AutoTokenizer.from_pretrained(self.src_model_name)
        self.tgt_tokenizer = AutoTokenizer.from_pretrained(self.tgt_model_name)

        self.batch_size = batch_size
        self.max_length = max_length
        # Issue #339: requesting a max_length larger than the model's positional
        # limit triggers a CUDA device-side assert
        # (indexSelectSmallIndex `srcIndex < srcSelectDimSize` failed).
        # Resolve the safe generation limit per model once at init time.
        self.src_max_length = self._resolve_model_max_length(self.src_model)
        self.tgt_max_length = self._resolve_model_max_length(self.tgt_model)
        self.max_length = self._clamp_max_length(max_length, self.src_max_length, self.tgt_max_length)

    def get_device(self):
        return str(self.src_model.device)

    @staticmethod
    def _resolve_model_max_length(model, default=512):
        """Return the largest safe generation length for a seq2seq model."""
        candidates = []
        config = getattr(model, 'config', None)
        if config is not None:
            for attr in ('max_length', 'max_position_embeddings', 'n_positions',
                         'max_target_positions', 'max_source_positions'):
                value = getattr(config, attr, None)
                if isinstance(value, int) and 0 < value < 100000:
                    candidates.append(value)
        generation_config = getattr(model, 'generation_config', None)
        if generation_config is not None:
            for attr in ('max_length', 'max_new_tokens'):
                value = getattr(generation_config, attr, None)
                if isinstance(value, int) and 0 < value < 100000:
                    candidates.append(value)
        if candidates:
            return min(candidates)
        return default

    @staticmethod
    def _clamp_max_length(requested, *limits):
        usable_limits = [limit for limit in limits if isinstance(limit, int) and limit > 0]
        if not usable_limits:
            return requested
        safe_limit = min(usable_limits)
        if requested is None:
            return safe_limit
        if requested > safe_limit:
            import warnings
            warnings.warn(
                f'max_length={requested} exceeds model limit {safe_limit}. '
                f'Clamping to {safe_limit} to avoid CUDA indexing errors (issue #339).'
            )
            return safe_limit
        return requested

    def _effective_max_length(self, model, model_max_length):
        if self.max_length is not None:
            if isinstance(model_max_length, int) and model_max_length > 0:
                return min(self.max_length, model_max_length)
            return self.max_length
        return model_max_length

    def predict(self, texts, target_words=None, n=1):
        src_translated_texts = self.translate_one_step_batched(texts, self.src_tokenizer, self.src_model)
        tgt_translated_texts = self.translate_one_step_batched(src_translated_texts, self.tgt_tokenizer, self.tgt_model)

        return tgt_translated_texts

    def translate_one_step_batched(
            self, data, tokenizer, model
    ):
        if not data:
            return []
        model_max_length = self._resolve_model_max_length(model)
        # Keep per-model limits up to date (e.g. after force_reload updates).
        if tokenizer is getattr(self, 'src_tokenizer', None):
            self.src_max_length = model_max_length
        elif tokenizer is getattr(self, 'tgt_tokenizer', None):
            self.tgt_max_length = model_max_length
        effective_max_length = self._effective_max_length(model, model_max_length)
        tokenized_texts = tokenizer(
            data, padding=True, truncation=True, max_length=effective_max_length,
            return_tensors='pt')
        if not tokenized_texts:
            return []

        all_translated_ids = []
        with torch.no_grad():
            item_count = next(iter(tokenized_texts.values())).shape[0]
            for start in range(0, item_count, self.batch_size):
                batch = {
                    key: value[start:start + self.batch_size].to(self.device)
                    for key, value in tokenized_texts.items()
                }
                if effective_max_length is None:
                    translated_ids_batch = model.generate(**batch)
                else:
                    translated_ids_batch = model.generate(**batch, max_length=effective_max_length)

                all_translated_ids.append(
                    translated_ids_batch.detach().cpu().numpy()
                )

        all_translated_texts = []
        for translated_ids_batch in all_translated_ids:
            translated_texts = tokenizer.batch_decode(
                translated_ids_batch,
                skip_special_tokens=True
            )
            all_translated_texts.extend(translated_texts)

        return all_translated_texts
