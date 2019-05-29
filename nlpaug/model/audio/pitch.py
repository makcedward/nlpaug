import librosa

from nlpaug.model.audio import Audio

"""
    Reference: https://www.kaggle.com/CVxTz/audio-data-augmentation
    A wrapper of librosa.effects.pitch_shift
"""

class Pitch(Audio):
    def __init__(self, sampling_rate, pitch_factor):
        super(Pitch, self).__init__()

        self.sampling_rate = sampling_rate
        self.pitch_factor = pitch_factor

    def manipulate(self, data):
        return librosa.effects.pitch_shift(data, self.sampling_rate, self.pitch_factor)
