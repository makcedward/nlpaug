try:
    import nltk
    from nltk.corpus import wordnet
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.word_dict import WordDictionary


class WordNet(WordDictionary):
    def __init__(self, lang, is_synonym=True):
        super().__init__(cache=True)

        self.lang = lang
        self.is_synonym = is_synonym
        self.model = self.read()

    def read(self):
        try:
            # Check whether wordnet package is downloaded
            wordnet.synsets('computer')
        except Exception:
            nltk.download('wordnet')

        try:
            # Check whether POS package is downloaded
            nltk.pos_tag('computer')
        except Exception:
            nltk.download('averaged_perceptron_tagger')

        return wordnet

    def predict(self, word, pos=None):
        results = []
        for synonym in self.model.synsets(word, pos=pos, lang=self.lang):
            for lemma in synonym.lemmas():
                if self.is_synonym:
                    results.append(lemma.name())
                else:
                    for antonym in lemma.antonyms():
                        results.append(antonym.name())
        return results

    def pos_tag(self, tokens):
        return nltk.pos_tag(tokens)
