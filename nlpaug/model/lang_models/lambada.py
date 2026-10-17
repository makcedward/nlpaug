import pandas as pd
import json
import os
try:
	import torch
	import torch.nn as nn
	from transformers import GPT2LMHeadModel, AutoTokenizer
	from simpletransformers.classification import ClassificationModel
except ImportError:
	# No installation required if not using this function
	pass

from nlpaug.model.lang_models import LanguageModels


class Lambada(LanguageModels):
	#https://arxiv.org/pdf/1911.03118.pdf
	def __init__(self, cls_model_dir, gen_model_dir, threshold=0.7, min_length=100, max_length=300, 
		batch_size=32, temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.0, device='cuda'):
		super().__init__(device, model_type=None)
		try:
			from transformers import GPT2LMHeadModel, AutoTokenizer
		except ModuleNotFoundError:
			raise ModuleNotFoundError('Missed transformers library. Install transfomers by `pip install transformers`')
		try:
			from simpletransformers.classification import ClassificationModel
		except ModuleNotFoundError:
			raise ModuleNotFoundError('Missed simpletransformers library. Install transfomers by `pip install simpletransformers`')

		self.cls_model_dir = cls_model_dir
		self.gen_model_dir = gen_model_dir
		self._validate_model_dirs(cls_model_dir, gen_model_dir)
		self.threshold = threshold
		self.min_length = min_length
		self.max_length = max_length
		self.batch_size = batch_size
		self.temperature = temperature
		self.top_k = top_k
		self.top_p = top_p
		self.repetition_penalty = repetition_penalty
		self.sep_token = '[SEP]'
		self.stop_token = '<|endoftext|>'

		with open(os.path.join(cls_model_dir, 'label_encoder.json')) as f:
			self.label2id = json.load(f)
			self.id2label = {v:k for k, v in self.label2id.items()}

		with open(os.path.join(cls_model_dir, 'cls_config.json')) as f:
			cls_config = json.load(f)
		self.cls_model = ClassificationModel(cls_config['model_type'], cls_model_dir, use_cuda=device != 'cpu', 
			args={'silent':True})
		self.gen_model = GPT2LMHeadModel.from_pretrained(gen_model_dir)
		self.gen_model.eval()
		self.gen_tokenizer = self._load_gen_tokenizer(gen_model_dir)
		self.to_device()

	@staticmethod
	def _validate_model_dirs(cls_model_dir, gen_model_dir):
		# Issue #335: users point gen_model_dir at the cls checkpoint (RoBERTa)
		# and get "tokenizer class ... is 'RobertaTokenizer' ... called from
		# 'GPT2Tokenizer'". Fail fast with an actionable message.
		if not cls_model_dir or not gen_model_dir:
			raise ValueError('Both cls_model_dir and gen_model_dir must be provided.')
		if os.path.abspath(cls_model_dir) == os.path.abspath(gen_model_dir):
			raise ValueError(
				'cls_model_dir and gen_model_dir must be different. '
				'Expected <model_dir>/cls and <model_dir>/gen from train_lambada.sh.')
		if os.path.isfile(os.path.join(gen_model_dir, 'cls_config.json')):
			raise ValueError(
				f"gen_model_dir '{gen_model_dir}' looks like a classification checkpoint "
				"(contains cls_config.json). Pass the GPT-2 generation checkpoint "
				"(<model_dir>/gen) instead (issue #335).")
		if not os.path.isfile(os.path.join(cls_model_dir, 'label_encoder.json')):
			raise ValueError(
				f"cls_model_dir '{cls_model_dir}' is missing label_encoder.json. "
				"Pass the classification checkpoint (<model_dir>/cls).")

	@staticmethod
	def _load_gen_tokenizer(gen_model_dir):
		from transformers import AutoTokenizer
		tokenizer = AutoTokenizer.from_pretrained(gen_model_dir)
		tokenizer_name = type(tokenizer).__name__
		# AutoTokenizer resolves the correct class; a RoBERTa/BERT tokenizer
		# here means the generation dir points at the wrong checkpoint.
		if 'Roberta' in tokenizer_name or 'Bert' in tokenizer_name:
			raise ValueError(
				f"Tokenizer loaded from '{gen_model_dir}' is '{tokenizer_name}', "
				"but LAMBADA generation needs a GPT-2 tokenizer. You likely passed "
				"the classification checkpoint (<model_dir>/cls) as gen_model_dir; "
				"use <model_dir>/gen instead (issue #335).")
		if getattr(tokenizer, 'pad_token', None) is None and getattr(tokenizer, 'eos_token', None) is not None:
			tokenizer.pad_token = tokenizer.eos_token
		return tokenizer

	@staticmethod
	def _strip_prompt(generated_text, input_text, sep_token):
		# Issue #328: blindly slicing len(input_text) chars mangles output when
		# BPE decoding inserts different spacing. Prefer structural stripping.
		if generated_text.startswith(input_text):
			return generated_text[len(input_text):]
		if sep_token and sep_token in generated_text:
			return generated_text.split(sep_token, 1)[1]
		return generated_text

	def to_device(self):
		self.gen_model.to(self.device)

	def get_device(self):
		return str(self.gen_model.device)

	# TODO: support gpt2 only
	def _generate(self, texts, n):
		results = []
		# Encode
		for label in texts:
			input_text = '{} {}'.format(label, self.sep_token)
			input_ids = self.gen_tokenizer.encode(input_text, add_special_tokens=False, return_tensors='pt')
			input_ids = input_ids.to(self.device)

			# Generate
			num_round, last_round = divmod(n, self.batch_size)

			for _n in [self.batch_size] * num_round + [last_round]:
				if _n == 0:
					break

				output_sequences = self.gen_model.generate(
					input_ids=input_ids,
					pad_token_id=50256, # avoid "Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation."
					min_length=self.min_length, 
					max_length=self.max_length + len(input_ids[0]),
					temperature=self.temperature,
					top_k=self.top_k,
					top_p=self.top_p,
					repetition_penalty=self.repetition_penalty,
					do_sample=True,
					num_return_sequences=_n,
				)

				if len(output_sequences.shape) > 2:
					output_sequences.squeeze_()

				# Decode
				for generated_sequence in output_sequences:
					generated_sequence = generated_sequence.tolist()
					# Decode text
					generated_text = self.gen_tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
					# Remove prompt prefix robustly (issue #328)
					generated_text = self._strip_prompt(generated_text, input_text, self.sep_token)
					# Remove all text after the stop token
					generated_text = generated_text[:generated_text.find(self.stop_token) if self.stop_token else None]
					# Replace sep_token by ' '
					generated_text = generated_text.replace(self.sep_token, ' ')

					results.append({'label': label, 'text': generated_text})
		
		results = pd.DataFrame(results).reset_index()
		results.rename(columns={'index': 'id'}, inplace=True)

		return results[['id', 'text', 'label']]

	def _classify(self, data):
		preds, raw_outputs = self.cls_model.predict(data['text'].tolist())

		probs = nn.functional.softmax(torch.from_numpy(raw_outputs), dim=1)
		probs = torch.max(probs, dim=1).values.tolist()
		probs = [round(p, 4) for p in probs]

		data['pred'] = [self.id2label[p] for p in preds]
		data['prob'] = probs
		return data

	def _filter(self, data):
		if not self.threshold:
			return data

		return data[
			(data['label'] == data['pred'])
			& (data['prob'] >= self.threshold)
		].sort_values(['label', 'pred'], ascending=[True, False])

	def predict(self, texts, target_words=None, n=10):
		generated_texts = self._generate(texts, n)
		generated_texts = self._classify(generated_texts)
		if self.threshold:
			return self._filter(generated_texts)
		return generated_texts
