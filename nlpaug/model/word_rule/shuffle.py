try:
	from nltk.tokenize import sent_tokenize
except ImportError:
	# No installation required if not using this function
	pass

from nlpaug.model.word_rule.word_rule import WordRule


class Shuffle(WordRule):
	TYPES = ['sentence']

	def __init__(self, model_type, mode='neighbor', tokenizer=None):
		super().__init__(cache=True)

		self.model_type = model_type # /sentence, word or character
		self.mode = mode

		if tokenizer:
			self.tokenizer = tokenizer
		else:
			if self.model_type == 'sentence':
				try:
					from nltk.tokenize import sent_tokenize
				except ModuleNotFoundError:
					raise ModuleNotFoundError('Missed nltk library. Install transfomers by `pip install nltk`')
				self.tokenizer = sent_tokenize

	def tokenize(self, data):
		return self.tokenizer(data)

	def predict(self, data, idx):
		if self.model_type == 'sentence': return self._predict_sentence(data, idx)

		return Exception(
			'{} is unexpected model_type. Possbile value is {}'.format(
				self.model_type, self.TYPES))

	def _predict_sentence(self, sentences, idx):
		last_idx = len(sentences) - 1
		direction = ''
		if self.mode == 'neighbor':
			if self.sample(2) == 0:
				direction = 'left'
			else:
				direction = 'right'
		if self.mode == 'left' or direction == 'left':
			if idx == 0:
				sentences[0], sentences[last_idx] = sentences[last_idx], sentences[0]
			else:
				sentences[idx], sentences[idx-1] = sentences[idx-1], sentences[idx]
		elif self.mode == 'right' or direction == 'right':
			if idx == last_idx:
				sentences[0], sentences[idx] = sentences[idx], sentences[0]
			else:
				sentences[idx], sentences[idx+1] = sentences[idx+1], sentences[idx]
		elif self.mode == 'random':
			idxes = self.sample(list(range(len(sentences))), num=2)
			for _id in idxes:
				if _id != idx:
					sentences[_id], sentences[idx] = sentences[idx], sentences[_id]
					break
		return sentences
