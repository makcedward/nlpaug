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

        try:
            import nltk
            from nltk.corpus import wordnet
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed nltk library. Install nltk by `pip install nltk`')

        # try:
        #     # Check whether wordnet package is downloaded
        #     wordnet.synsets('computer')
        #     # Check whether POS package is downloaded
        #     nltk.pos_tag('computer')
        # except LookupError:
        #     nltk.download('wordnet')
        #     nltk.download('averaged_perceptron_tagger')

        self.model = self.read()

    def read(self):
        return wordnet

    def predict(self, word, pos=None):
        results = []
        for synonym in self.model.synsets(word, pos=pos, lang=self.lang):
            for lemma in synonym.lemmas(lang=self.lang):
                if self.is_synonym:
                    results.append(lemma.name())
                else:
                    for antonym in lemma.antonyms():
                        results.append(antonym.name())
        return results

    @classmethod
    def pos_tag(cls, tokens):
        return nltk.pos_tag(tokens)
