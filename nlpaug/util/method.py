class Method:
    CHAR = 'char'
    WORD = 'word'
    SPECTROGRAM = 'spectrogram'

    FLOW = 'flow'

    @staticmethod
    def getall():
        return [Method.CHAR, Method.WORD, Method.SPECTROGRAM, Method.FLOW]

