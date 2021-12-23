try:
    import torch
    from torch.utils import data as t_data
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
        self.src_model = AutoModelForSeq2SeqLM.from_pretrained(self.src_model_name)
        self.src_model.eval()
        self.src_model.to(device)
        self.tgt_model = AutoModelForSeq2SeqLM.from_pretrained(self.tgt_model_name)
        self.tgt_model.eval()
        self.tgt_model.to(device)
        self.src_tokenizer = AutoTokenizer.from_pretrained(self.src_model_name)
        self.tgt_tokenizer = AutoTokenizer.from_pretrained(self.tgt_model_name)

        self.batch_size = batch_size
        self.max_length = max_length

    def get_device(self):
        return str(self.src_model.device)

    def predict(self, texts, target_words=None, n=1):
        src_translated_texts = self.translate_one_step_batched(texts, self.src_tokenizer, self.src_model)
        tgt_translated_texts = self.translate_one_step_batched(src_translated_texts, self.tgt_tokenizer, self.tgt_model)

        return tgt_translated_texts

    def translate_one_step_batched(
            self, data, tokenizer, model
    ):
        tokenized_texts = tokenizer(data, padding=True, return_tensors='pt')
        tokenized_dataset = t_data.TensorDataset(*(tokenized_texts.values()))
        tokenized_dataloader = t_data.DataLoader(
            tokenized_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1
        )

        all_translated_ids = []
        with torch.no_grad():
            for batch in tokenized_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask = batch

                translated_ids_batch = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_length=self.max_length
                )

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
