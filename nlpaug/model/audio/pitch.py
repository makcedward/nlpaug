import librosa
import numpy as np

from nlpaug.model.audio import Audio

"""
    Reference: https://www.kaggle.com/CVxTz/audio-data-augmentation
    A wrapper of librosa.effects.pitch_shift
"""

class Pitch(Audio):
    def __init__(self, sampling_rate, pitch_factor):
        """

        :param sampling_rate: Sampling rate of input audio
        :param pitch_factor: Number of half-steps that shifting audio
        """
        super(Pitch, self).__init__()

        self.sampling_rate = sampling_rate
        self.pitch_factor = pitch_factor

    def manipulate(self, data):
        n_step = np.random.randint(self.pitch_factor[0], self.pitch_factor[1])

        return librosa.effects.pitch_shift(data, self.sampling_rate, n_step)
