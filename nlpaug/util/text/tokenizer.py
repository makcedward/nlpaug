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

# Issue #231: joining CJK characters with spaces corrupts Chinese output
# ("咋就不行了？" -> "咋 就 不 行 了 ？"). Strip spaces between CJK
# characters and before CJK punctuation after the generic join.
CJK_CHAR = r'\u4e00-\u9fff\uf900-\ufaff\u3400-\u4dbf\U00020000-\U0002a6df'
CJK_PUNCT = r'\u3000-\u303f\uff00-\uffef'
CJK_DETOKENIZER_REGEXS = [
	(re.compile(f'(?<=[{CJK_CHAR}{CJK_PUNCT}]) (?=[{CJK_CHAR}{CJK_PUNCT}])'), ''),
	(re.compile(f'(?<=[{CJK_CHAR}]) (?=[{CJK_PUNCT}])'), ''),
]

SENTENCE_SEPARATOR = '.!?'

SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+(?=[\"\'\(\[]?[A-Z0-9])|(?<=[.!?])\s*$')


def add_space_around_punctuation(text):
    return ADDING_SPACE_AROUND_PUNCTUATION_REGEX.sub(r' ', text)


def split_sentence(text):
    return SPLIT_WORD_REGEX.findall(text)


def split_sentences(text):
    """Split long document into sentences (issue #336).

    Keeps sentence-ending punctuation attached. Falls back to the whole
    stripped text when no sentence boundary is found.
    """
    if not text or not text.strip():
        return []
    parts = SENTENCE_SPLIT_REGEX.split(text.strip())
    sentences = [p.strip() for p in parts if p and p.strip()]
    if not sentences:
        return [text.strip()]
    # re.split with trailing `\s*$` can yield empty tail; already filtered.
    return sentences


def split_long_document(text, max_sentences=5, max_tokens=None):
    """Chunk a long document into augmenter-friendly pieces (issue #336).

    :param str text: Long document text.
    :param int max_sentences: Max sentences per chunk. Must be >= 1.
    :param int max_tokens: Optional max whitespace-separated tokens per chunk.
        When set, chunks are further split so no chunk exceeds it.
    :return: List of chunk strings. Empty input returns [].
    """
    if not text or not text.strip():
        return []
    if max_sentences is None or max_sentences < 1:
        raise ValueError('max_sentences must be >= 1')
    if max_tokens is not None and max_tokens < 1:
        raise ValueError('max_tokens must be >= 1 or None')

    sentences = split_sentences(text)
    # No sentence boundary (e.g. single long paragraph): fall back to
    # token-window chunking so model-based augmenters still get bounded input.
    if len(sentences) <= 1 and max_tokens is not None:
        words = text.strip().split()
        return [' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk_sentences = sentences[i:i + max_sentences]
        if max_tokens is None:
            chunks.append(' '.join(chunk_sentences))
        else:
            current, current_len = [], 0
            for sent in chunk_sentences:
                sent_len = len(sent.split())
                if current and current_len + sent_len > max_tokens:
                    chunks.append(' '.join(current))
                    current, current_len = [sent], sent_len
                else:
                    current.append(sent)
                    current_len += sent_len
            if current:
                chunks.append(' '.join(current))
    return chunks

class Tokenizer:
	@staticmethod
	def tokenizer(text):
		tokens = TOKENIZER_REGEX.split(text)
		return [t for t in tokens if len(t.strip()) > 0]

	@staticmethod
	def reverse_tokenizer_with_original(tokens, original_text):
		token_iter = iter(tokens)
		parts = TOKENIZER_REGEX.split(original_text)
		return ''.join(next(token_iter) if part.strip() else part for part in parts)

	@staticmethod
	def reverse_tokenizer(tokens):
		text = ' '.join(tokens)
		for regex, sub in DETOKENIZER_REGEXS:
			text = regex.sub(sub, text)
		for regex, sub in CJK_DETOKENIZER_REGEXS:
			text = regex.sub(sub, text)
		return text.strip()
