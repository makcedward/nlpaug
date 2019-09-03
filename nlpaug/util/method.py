class Method:
    CHAR = 'char'
    WORD = 'word'
    SENTENCE = 'sentence'
    SPECTROGRAM = 'spectrogram'
    AUDIO = 'audio'

    FLOW = 'flow'

    @staticmethod
    def getall():
        return [Method.CHAR, Method.WORD, Method.SENTENCE, Method.AUDIO, Method.SPECTROGRAM, Method.FLOW]

