import re

ADDING_SPACE_AROUND_PUNCTUATION_REGEX = re.compile(r'(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )')
SPLIT_WORD_REGEX = re.compile(r'\b.*?\S.*?(?:\b|$)')

# re.compile(r"(\W+)")
# re.compile(r"\w+|[^\w\s]")


def add_space_around_punctuation(text):
    return ADDING_SPACE_AROUND_PUNCTUATION_REGEX.sub(r' ', text)


def split_sentence(text):
    return SPLIT_WORD_REGEX.findall(text)

