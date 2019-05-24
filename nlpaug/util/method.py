class Method:
    CHAR = 'char'
    WORD = 'word'
    SPECTROGRAM = 'spectrogram'
    AUDIO = 'audio'

    FLOW = 'flow'

    @staticmethod
    def getall():
        return [Method.CHAR, Method.WORD, Method.AUDIO, Method.SPECTROGRAM, Method.FLOW]

