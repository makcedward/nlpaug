import re

ADDING_SPACE_AROUND_PUNCTUATION_REGEX = re.compile(r'(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )')

SPLIT_WORD_REGEX = re.compile(r'\b.*?\S.*?(?:\b|$)')

TOKENIZER_REGEX = re.compile(r'(\W)')
DETOKENIZER_REGEXS = [
	(re.compile(r'\s([.,:;?!%]+)([ \'"`])'), r'\1\2'), # End of sentence
	(re.compile(r'\s([.,:;?!%]+)$'), r'\1'), # End of sentence
	(re.compile(r'\s([\[\(\{\<])\s'), r' \g<1>'), # Left bracket
	(re.compile(r'\s([\]\)\}\>])\s'), r'\g<1> '), # right bracket
]

SENTENCE_SEPARATOR = '.!?'

def add_space_around_punctuation(text):
    return ADDING_SPACE_AROUND_PUNCTUATION_REGEX.sub(r' ', text)

def split_sentence(text):
    return SPLIT_WORD_REGEX.findall(text)

class Tokenizer:
	@staticmethod
	def tokenizer(text):
		tokens = TOKENIZER_REGEX.split(text)
		return [t for t in tokens if len(t.strip()) > 0]

	@staticmethod
	def reverse_tokenizer(tokens):
		text = ' '.join(tokens)
		for regex, sub in DETOKENIZER_REGEXS:
			text = regex.sub(sub, text)
		return text.strip()